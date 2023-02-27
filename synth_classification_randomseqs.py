# code to run cycles of synthetic scoring prediction from randomized 300bp seqs
# with data reduction and 5-fold CV

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
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import time
from torch.utils.data.sampler import WeightedRandomSampler

import models as m
import utils as u
import torch_utils as tu
from torch_utils import DatasetSpec

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        'A':5,
        'C':2,
        'G':-2,
        'T':-5
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

def set_reg_class_up_down(df, col,thresh=1.0):
    '''
    Given a dataframe of log ratio TPMS, add a column splitting genes into categories
    * Below -thresh: class 0
    * Between -thresh:thresh: class 1
    * Above thresh: class 2
    '''
    def get_class(val):
        if val < -thresh:
            return 0
        elif val > thresh:
            return 2
        else:
            return 1
    
    reg_col = f"{col}_reg_UD"
    df[reg_col] = df[col].apply(lambda x: get_class(x))

def make_weighted_sampler(df, target_col):
    '''
    Given a training dataframe, create a balanced sampler for the class
    indicated
    '''
    # make weighted sampler for data loader
    class_sample_count = df[target_col].value_counts()
    # get 1/count as weight for each class
    weight = dict([(x,(1. / class_sample_count[x])) for x in class_sample_count.keys()])
    # apply new weight to each sample
    samples_weight = np.array([weight[t] for t in df[target_col].values])
    samples_weight = torch.from_numpy(samples_weight).double()

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


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
            3, # num tasks
        )
    model.to(DEVICE)

    # currently hardcoded for classification
    #loss_func = torch.nn.MSELoss() 
    loss_func = torch.nn.CrossEntropyLoss()
    
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

def get_confusion_stats(model,model_name,seq_list,title=None,save_file=True):#seqs,labels,seq_name):
    '''Get class predictions and plot confusion matrix'''

    def plot_confusion_raw_norm(mats):
        f, axes = plt.subplots(len(seq_list), 2, figsize=(9.8, 4.2*len(seq_list)))#, sharey='row')
        #axes = list(axes)
        axes_list = [item for sublist in axes for item in sublist]

        for i,(mat,subtitle) in enumerate(mats):
            disp = ConfusionMatrixDisplay(confusion_matrix=mat)
            disp.plot(ax=axes_list.pop(0))
            #disp.plot(ax=axes.pop(0))
            disp.ax_.set_title(f"{subtitle}")
            disp.im_.colorbar.remove()

        title_str=title if title else model_name
        f.suptitle(f"{title_str}",fontsize=20)
        plt.tight_layout()
        if save_file:
            plt.savefig(save_file)

    model.eval()
    print(f"Running {model_name}")
    
    mats = [] # conf matrices
    res_data = [] # classification results

    for seqs, labels, split_name in seq_list:
        ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs]).to(DEVICE)
        preds = [x.topk(1).indices.item() for x in model(ohe_seqs.float())]#.tolist()        
        
        cls_rep = classification_report(labels, preds,output_dict=True)
        pr = cls_rep['macro avg']['precision']
        re = cls_rep['macro avg']['recall']
        f1 = cls_rep['macro avg']['f1-score']
        sp = cls_rep['macro avg']['support']
        res_data.append([model_name,split_name,pr,re,f1,sp])
        
        c = confusion_matrix(labels, preds)
        mats.append((c,f"raw counts ({split_name})"))
        # get the normalized confusino matrix
        cp = np.zeros(c.shape)
        for i,row in enumerate(c):
            rowsum = sum(row)
            for j,item in enumerate(row):
                val = item/rowsum
                cp[i][j] = val

        mats.append((cp,f"normed counts ({split_name})"))

    plot_confusion_raw_norm(mats)
    
    res_df = pd.DataFrame(res_data,columns=['model_name','split','mac_precision','mac_recall','mac_f1','support'])
    
    return res_df
    


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
            3, # num tasks
            num_filters1=8,
            num_filters2=4,
            kernel_size1=8,
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
            num_classes=3
        )
    elif model_type == 'CNNLSTM':
        model = m.DNA_CNNLSTM(
            input_size,
            DEVICE,
            num_classes=3,
            num_filters=8,
            kernel_size=10,
            fc_node_num1=10
        )

    elif model_type == 'CNN_simple':
        model = m.DNA_CNN(
            input_size,
            num_filters=8,
            kernel_size=8,
            num_classes=3,
            fc_node_num=5
        )

    else:
        raise ValueError(f"Unknown model type {model_type}. (Current: CNN, biLSTM, CNNLSTM)")

    return model


def main():
    set_seed()

    # # load 5G TPM data
    # data_filename = "data/XY_lr_noCu_opFilt_20221031.tsv"
    # XYdf_og = pd.read_csv(data_filename,sep='\t')

    # # remove seq with N's for now
    # XYdf = XYdf_og[~XYdf_og['upstream_region'].str.contains("N")]

    # # add synthetic score column
    # XYdf['score'] = XYdf['upstream_region'].apply(lambda x: synthetic_score(x))

    # # add synthetic classification category
    # set_reg_class_up_down(XYdf,'score',thresh=5)


    # +----------------------------------------------+
    # TODO load info from config file
    cvs = [0,1,2,3,4]
    #augs = [0]
    reductions = [0.1,0.25,0.5]
    models_to_try = ['CNN','CNN_simple']
    out_dir = 'out_synth_cls_randomseq_rw_5fold' #'pred_out'

    seq_col_name = 'upstream_region' # TODO: put in config
    target_col_name = 'score_reg_UD' # TODO: put in config
    locus_col_name = 'locus_tag'

    reweight_samples = True 
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
        train_df_og = pd.read_csv(f'data/synth_cls_randomseq_splits/cv{fold}_train_12000.tsv',sep='\t')
        test_df_og = pd.read_csv(f'data/synth_cls_randomseq_splits/cv{fold}_test_12000.tsv',sep='\t')

        # for each round of reduction
        for r in reductions:
            # reduce the train/test size
            train_df = train_df_og.sample(frac=r)
            test_df = test_df_og.sample(frac=r)
            train_size = train_df.shape[0]

            split_dfs = {
                #'full_train':full_train_df,
                'train':train_df,
                #'val':val_df,
                'test':test_df,   
            }

            dataset_types = [
                DatasetSpec('ohe'),
            ]

            
            seq_len = len(train_df[seq_col_name].values[0])
            sampler = make_weighted_sampler(train_df,target_col_name) if  reweight_samples else None
            # put into pytorch dataloaders
            dls = tu.build_dataloaders_single(
                train_df, 
                test_df, 
                dataset_types, # just OHE for now
                seq_col=seq_col_name,
                target_col=target_col_name,
		        sampler=sampler,
                shuffle=False if reweight_samples else True,
            )
            ohe_train_dl,ohe_val_dl = dls['ohe']

            # for each type of model
            for model_type in models_to_try:
                print(f"running model {model_type} for red={r}X (CVfold {fold})")
                # model + reduction combo
                combo_name = f"{model_type}_r{r}X_cv{fold}" 
            
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
                tu.quick_loss_plot(t_res['data_label'],title=f"{combo_name} Loss Curve",save_file=f"{out_dir}/{combo_name}_loss_plot.png")

                # confusion matrix plotting
                seq_list = [
                    (train_df[seq_col_name].values,train_df[target_col_name],"train"),
                    (test_df[seq_col_name].values,test_df[target_col_name],"test")
                ]
                p_res_df = get_confusion_stats(
                    model,
                    model_name,
                    seq_list,
                    save_file=f"{out_dir}/{combo_name}_confmat.png",
                    title=f"{combo_name}"
                )

                p_res_df['reduction'] = r
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
