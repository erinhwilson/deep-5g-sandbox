# code to run cycles of synthetic scoring prediction from 5G sequences
# with data augmentation and 5-fold CV

import torch
from torch import nn

import altair as alt
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import time

import models as m
import utils as u
import torch_utils as tu
from torch_utils import DatasetSpec

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')
print(DEVICE)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    #os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def synthetic_score(seq):
    '''
    Given a DNA sequence, return a simple synthetic score based on
    it's sequence content and presence of a specific 6mer motif
    '''
    score_dict = {
        'A':10,
        'C':7,
        'G':4,
        'T':1
    }

    score = np.mean([score_dict[base] for base in seq])
    if 'TATATA' in seq:
        score += 10
    if 'GCGCGC' in seq:
        score -= 10
    return score

def augment_mutate(df,n,seq_col='upstream_region',mutation_rate=0.03):
    '''
    Given a dataframe of training data, augment it by adding 
    mutated versions back into the data frame
    '''
    mutation_dict = {
        'A':['C','G','T'],
        'C':['G','T','A'],
        'G':['T','A','C'],
        'T':['A','C','G']
    }
    # init seq_version column
    df['seq_version'] = 0

    # no augmentation requested
    if n == 0:
        return df
    
    # augment by adding n more copies
    else:
        new_rows = []
        # for each row in the original df
        for i,row in df.iterrows():
            seq = row[seq_col]
                    
            # generate n mutants
            for j in range(n):
                new_row = copy.deepcopy(row)
                new_seq = list(seq)
                mutate_vec = [random.random() for x in range(len(seq))]
                
                # loop through mutation values along length of the seq
                for k in range(len(seq)):
                    # if random value is below mutation rate, then make a change
                    if mutate_vec[k] < mutation_rate:
                        cur_base = seq[k]
                        # select new base randomly
                        new_base = random.choice(mutation_dict[cur_base])
                        new_seq[k] = new_base
                
                new_row[seq_col] = ''.join(new_seq)
                new_row['seq_version'] = j+1
                new_rows.append(new_row.values)

        # put new rows into a df
        new_rows = pd.DataFrame(new_rows,columns=new_row.index)        
        
        return pd.concat([df,new_rows])


def collect_model_stats(model_name,seq_len,
                        train_dl,val_dl,
                        lr=0.001,ep=1000,pat=100,
                        opt=None,model=None,load_best=True,chkpt_path='checkpoint.pt'):
    '''
    Execute run of a model and return stats and objects related
    to its results
    '''
    # default model if none specified
    if not model:
        model = m.DNA_2CNN_2FC_Multi(
            seq_len,
            1, # num tasks
        )
    model.to(DEVICE)

    # currently hardcoded for regression
    loss_func = torch.nn.MSELoss() 
    
    if opt:
        opt = opt(model.parameters(), lr=lr)

    # collect run time
    start_time = time.time()
    
    train_losses, \
    val_losses, \
    epoch_stop, \
    best_val_score = tu.run_model(
        train_dl,
        val_dl, 
        model, 
        loss_func, 
        DEVICE,
        lr=lr, 
        epochs=ep, 
        opt=opt,
        patience=pat,
        load_best=load_best,
        chkpt_path=chkpt_path
    )
    total_time = time.time() - start_time

    # to plot loss
    data_label = [((train_losses,val_losses),model_name,epoch_stop,best_val_score)]
    #tu.quick_loss_plot(data_label)
    
    return {
        'model_name':model_name,
        'model':model,
        'train_losses':train_losses,
        'val_losses':val_losses,
        'epoch_stop':epoch_stop,
        'best_val_score':best_val_score,
        'data_label':data_label,
        'total_time':total_time
    }

def parity_pred_by_split(model,
                         model_name,
                         device,
                         split_dfs,
                         locus_col='locus_tag',
                         seq_col='seq',
                         target_col="score",
                         splits=['train','val'],
                         alpha=0.2,
                         save_file=None
                        ):
    '''
    Given a trained model, get the model's predictions on each split
    of the data and create parity plots of the y predictions vs actual ys
    '''
    # init subplots
    fig, axs = plt.subplots(1,len(splits), sharex=True, sharey=True,figsize=(10,4))
    #pred_dfs = {}
    pred_res = [] # collect prediction results for dataFrame
    
    def parity_plot(title,ytrue,ypred,rigid=True):
        '''
        Individual parity plot for a specific split
        '''
        axs[i].scatter(ytrue, ypred, alpha=alpha)

        r2 = r2_score(ytrue,ypred)
        pr = pearsonr(ytrue,ypred)[0]
        sp = spearmanr(ytrue,ypred).correlation

        # y=x line
        xpoints = ypoints = plt.xlim()
        if rigid:
            axs[i].set_ylim(min(xpoints),max(xpoints)) 
        axs[i].plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False, scaley=False)
        axs[i].set_title(f"{title} (r2:{r2:.2f}|p:{pr:.2f}|sp:{sp:.2f})",fontsize=14)
        axs[i].set_xlabel("Actual Score",fontsize=14)
        axs[i].set_ylabel("Predicted Score",fontsize=14)

        return r2, pr, sp
    
    for i,split in enumerate(splits):
        print(f"{split} split")
        df = split_dfs[split]
        loci = df[locus_col].values
        seqs = list(df[seq_col].values)        
        ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs]).to(device)
        labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)
    
    #dfs = {} # key: model name, value: parity_df
    
        # initialize prediction df with just locus col
        pred_df = df[[locus_col]]
        pred_df['truth'] = df[target_col]
        print(f"Predicting for {model_name}")
        
        
        # ask model to predict on seqs
        preds = model(ohe_seqs.float()).tolist()
        # preds is a tensor converted to a list, 
        # single elements returned as a list, hence x[0]
        pred_df['pred'] = [x[0] for x in preds]
        
        # do I want the result dfs? revise if so
        #dfs[model_name] = pred_df
        
        # plot stuff
        ytrue = pred_df['truth'].values
        ypred = pred_df['pred'].values
        
        #plt.subplot(len(splits),i+1,1)
        split_title = split
        r2,pr,sp = parity_plot(split_title,ytrue,ypred,rigid=True)
        
        # save predictions
        #pred_dfs[split] = pred_df
        row = [model_name,split,r2,pr,sp]
        pred_res.append(row)
    
    # show combined plot
    plt.suptitle(model_name,fontsize=14)
    plt.tight_layout()
    plt.show()
    if save_file:
        plt.savefig(save_file)
    
    return pd.DataFrame(pred_res, columns=['model_name','split','r2','pearson','spearman'])



def quick_model_setup(model_type,input_size):
    '''
    Some quick model types - make customizable later
    '''
    if model_type == 'CNN':
        # model = m.DNA_2CNN_2FC_Multi(
        #     input_size,
        #     1, # num tasks
        # )

        # start smaller model for synth task
        model = m.DNA_2CNN_2FC_Multi(
            input_size,
            1, # num tasks
            num_filters1=8,
            num_filters2=4,
            kernel_size1=10,
            kernel_size2=6,
            conv_pool_size1=2,
            fc_node_num1=5,
            fc_node_num2=5,
            dropout1=0.25
        )

    elif model_type == 'biLSTM':
        model = m.DNA_biLSTM(
            input_size,
            DEVICE,
            num_classes=1
        )
    elif model_type == 'CNNLSTM':
        model = m.DNA_CNNLSTM(
            input_size,
            DEVICE,
            num_classes=1,
            num_filters=8,
            kernel_size=10,
            fc_node_num1=10
        )
    elif model_type == 'CNN_simple':
        model = m.DNA_CNN(
            input_size,
            num_filters=8,
            kernel_size=10,
            num_classes=1
        )

    else:
        raise ValueError(f"Unknown model type {model_type}. (Current: CNN, biLSTM, CNNLSTM, CNN_simple)")

    return model


def main():
    set_seed()

    # load 5G TPM data
    data_filename = "data/XY_logTPM_opFilt_20221031.tsv"
    XYdf_og = pd.read_csv(data_filename,sep='\t')

    # remove seq with N's for now
    XYdf = XYdf_og[~XYdf_og['upstream_region'].str.contains("N")]

    # add synthetic score column
    XYdf['score'] = XYdf['upstream_region'].apply(lambda x: synthetic_score(x))

    # +----------------------------------------------+
    # TODO load info from config file
    cvs = [0,1,2,3,4]
    #cvs=[4] 
    #augs = [0,10,50,100]
    augs = [0,10]
    #models_to_try = ['CNN','CNNLSTM','biLSTM']
    models_to_try = ['CNN_simple']
    out_dir = 'out_synth_reg_5fold' #'pred_out'

    seq_col_name = 'upstream_region' # TODO: put in config
    target_col_name = 'score' # TODO: put in config
    locus_col_name = 'locus_tag'
    # +----------------------------------------------+

    # make out and checkpoint dirs
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Making out_dir:",out_dir)
    chkpt_dir = os.path.join(out_dir, "chkpts")
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
        print("Making chkpt_dir:",chkpt_dir)

    # init result collection objects
    training_res = {}               # training results
    all_pred_res = pd.DataFrame()   # prediction results

    # load the pre-made GroupShuffle splits (keep similar promoters in same split)
    for fold in cvs:
        train_df = pd.read_csv(f'data/synth_score_splits/cv{fold}_train.tsv',sep='\t')
        val_df = pd.read_csv(f'data/synth_score_splits/cv{fold}_val.tsv',sep='\t')
        test_df = pd.read_csv(f'data/synth_score_splits/cv{fold}_test.tsv',sep='\t')

        # for each round of augmentation
        for a in augs:
            print(f"aug = {a}X")
            # augment the train dataset size
            train_df_aug = augment_mutate(train_df,a,seq_col=seq_col_name)
            train_size = train_df_aug.shape[0]

            split_dfs = {
                #'full_train':full_train_df,
                'train':train_df_aug,
                'val':val_df,
                'test':test_df,   
            }

            dataset_types = [
                DatasetSpec('ohe'),
            ]

            
            seq_len = len(train_df_aug[seq_col_name].values[0])

            # put into pytorch dataloaders
            dls = tu.build_dataloaders_single(
                train_df_aug, 
                val_df, 
                dataset_types, # just OHE for now
                seq_col=seq_col_name,
                target_col=target_col_name,
                #batch_size=10, # UNDO: make smaller batch size?
            )
            ohe_train_dl,ohe_val_dl = dls['ohe']

            # for each type of model
            for model_type in models_to_try:
                print(f"running model {model_type} for aug={a}X (CVfold {fold})")
                # model + reduction combo
                combo_name = f"{model_type}_aug{a}X_cv{fold}" 
            
                # initialize a model
                model = quick_model_setup(model_type,seq_len)
                # TODO: generate model name from something about model spec
                model_name = model_type
                
                # run model and collect stats from training
                t_res = collect_model_stats(
                    model_name,
                    seq_len,
                    ohe_train_dl,
                    ohe_val_dl,
                    lr=0.0001,
                    ep=5000,
                    pat=500,
                    opt=torch.optim.Adam,
                    model=model,
                    chkpt_path=os.path.join(chkpt_dir,f"{combo_name}_chkpt.pt")
                )
                # save this in training res
                training_res[combo_name] = t_res # does this go anywhere? get saved?
                # plot loss 
                tu.quick_loss_plot(t_res['data_label'],save_file=f"{out_dir}/{combo_name}_loss_plot.png")

                splits_to_plot = ['val','test'] if train_size > 10000 else ['train','val','test']
                # collect prediction stats
                p_res_df = parity_pred_by_split(
                    model,
                    model_name,
                    DEVICE,
                    split_dfs,
                    locus_col=locus_col_name,
                    seq_col=seq_col_name,
                    target_col=target_col_name,
                    splits=splits_to_plot,
                    save_file=f"{out_dir}/{combo_name}_parity_plot.png"
                )
                p_res_df['augmentation'] = a
                p_res_df['train_size'] = train_size
                p_res_df['cv_fold'] = fold # which cv split
                p_res_df['best_val_score'] = t_res['best_val_score']
                p_res_df['epoch_stop'] = t_res['epoch_stop']
                p_res_df['total_time'] = t_res['total_time']
                pred_file_name = f"{out_dir}/{combo_name}_pred_res.tsv"
                # save a temp copy of the results?
                p_res_df.to_csv(pred_file_name,sep='\t',index=False)
                
                # add to master collection df
                all_pred_res = pd.concat([all_pred_res,p_res_df]) # add whole data row to final collection


    # after all this save the whole df
    all_pred_res.to_csv(f"{out_dir}/all_pred_res.tsv",sep='\t',index=False)
    
    print("DONE!")

if __name__ == '__main__':
    main()
