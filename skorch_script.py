import torch
from torch import nn
from torch.nn import functional as F

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import scipy

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform,uniform
#from random import randint

from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping,Checkpoint

import utils as u 
import torch_utils as tu
import models as m 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:",DEVICE)

def load_data(config):
    id_col = config['id_col']
    seq_col = config['seq_col']

    # make locus2info dict
    locus_info_file = config['locus_info_file']
    locus_info_df = pd.read_csv(locus_info_file,sep='\t')
    locus2info = u.make_info_dict(locus_info_df)

    # load TPM tsv and add info
    data_filename = config['expression_file']
    XY_df = pd.read_csv(data_filename,sep='\t')
    XY_df['gene'] = XY_df[id_col].apply(lambda x: locus2info[x]['gene'])
    XY_df['product'] = XY_df[id_col].apply(lambda x: locus2info[x]['product'])

    # dict of locus to sequence
    loc2seq = dict([(x,z) for (x,z) in XY_df[[id_col,seq_col]].values])

    condition_file = config['condition_file']
    with open(condition_file,'r') as f:
        conditions = list(x.strip() for x in f.readlines())
        
    cond_dict = dict(enumerate(conditions))
    cond_dict

    return locus2info, XY_df, loc2seq, conditions, cond_dict


def get_params():
    # params specific for CNN of a certain type:
        # current: models.py DNA_2CNN
    params = {
        'lr': [0.0005, 0.0001,0.00005,0.00001],#loguniform(0.0001, 0.01)
        
        'module__num_filters1': [16,32,64,128], # uniform(8,128), #
        'module__num_filters2': [16,32,64,128],
        
        'module__kernel_size1': [6,8,12,16,32],
        'module__kernel_size2': [6,8,12,16,32],
        
        'module__conv_pool_size1': [1,3,6,12],
        'module__fc_node_num1': [10, 25, 50,100], #randint(10,100), #
        
        'module__dropout1': [0.0,0.2,0.5],
        'module__dropout2': [0.0,0.2,0.5],
        
        'optimizer':[torch.optim.SGD, torch.optim.Adam, torch.optim.Adagrad,torch.optim.AdamW,torch.optim.RMSprop]
    }

    return params 

def setup_config():

    config = {
        # data inputs
        'expression_file':'data/XY_logTPM_opFilt.tsv',
        'locus_info_file':'data/locus2info.tsv',
        'condition_file':'data/conditions_to_include.txt',

        # outputs
        'out_dir':'skorch_test_noCu',
        'job_name':'skorch_randcv_st_noCu',

        # data specifics
        'id_col':'locus_tag',
        'seq_col':'upstream_region',
        'target_col':'NoCu',
        
        # model specifics
        'model_type':'2CNN',
        'skorch_params':get_params(),
        'epochs':5000, 
        'patience':500,

        # skorch search specifics
        'search_iters':1000, 
        
    }

    return config


def make_st_skorch_dfs(df,seq_col='seq',target_col='score'):
    '''
    Make basic X,y matrix,vec for skorch fit() loop.
    '''
    seqs = list(df[seq_col].values)        
    ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs])

    labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)
    # had to unsqueeze here or else errors later
    
    return ohe_seqs.float(), labels.float()


def make_mt_skorch_dfs(df,seq_col='seq',target_cols=['highCu','noCu']):
    '''
    Make multi-task X,y matrix,vec for skorch fit() loop.
    '''
    seqs = list(df[seq_col].values)        
    ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs])

    # number of labels = len(target_cols)
    labels = torch.tensor(list(df[target_cols].values))
    # bad dimension? fixed in model.forward for now
    
    return ohe_seqs.float(), labels.float()


def get_model_choice(choice):
    choices = ['LinearDeep', 'CNN', '2CNN', 'LSTM','CNNLSTM','Kmer']
    
    # LINEAR
    if choice == 'LinearDeep':
        return m.DNA_Linear_Deep

    # CNN 
    elif choice == "CNN":
        return m.DNA_CNN
    elif choice == "2CNN":
        return m.DNA_2CNN

    # LSTM
    elif choice == "LSTM":
        return m.DNA_LSTM

    # CNN-LSTM
    elif choice == "CNNLSTM":
        return m.DNA_CNNLSTM

    # Kmer 
    elif choice == "Kmer":
        return m.Kmer_Linear

    else:
        raise ValueError(f"{choice} model choice not recognized. Options are: {choices}")

def quick_loss_plot_simple(data_label_list,out_file,loss_type="MSE Loss",sparse_n=0):
    '''
    For each train/test loss trajectory, plot loss by epoch
    '''
    fig = plt.figure()
    for i,((train_data,test_data),label) in enumerate(data_label_list):
        # plot only 1 in every sparse_n points
        if sparse_n:
            train_data = [x for i,x in enumerate(train_data) if (i%sparse_n==0)]
            test_data = [x for i,x in enumerate(test_data) if (i%sparse_n==0)]
            
        plt.plot(train_data,'--',color=f"C{i}", label=f"{label} Train")
        plt.plot(test_data,'o-',color=f"C{i}", label=f"{label} Test",linewidth=3.0)
        

    plt.legend()
    plt.ylabel(loss_type)
    plt.xlabel("Epoch")
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.savefig(out_file,bbox_inches='tight')

def parity_plot(model_name,ytrue,ypred, pearson,rigid=False, out_dir="out_dir"):
    
    fig = plt.figure()
    plt.scatter(ytrue, ypred, alpha=0.2)
    
    # y=x line
    xpoints = ypoints = plt.xlim()
    if rigid:
        plt.ylim(min(xpoints),max(xpoints)) 
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False, scaley=False)

    plt.xlabel("Actual Score",fontsize=14)
    plt.ylabel("Predicted Score",fontsize=14)
    plt.title(f"{model_name} (pearson:{pearson:.3f})",fontsize=20)
    plt.savefig(f'{out_dir}/{model_name}_parity_plot.png',bbox_inches='tight')


# #####################################################
def main():

    # +----------------------+
    # | Load data and config |
    # +----------------------+
    config = setup_config()
    id_col = config['id_col']
    seq_col = config['seq_col']
    target_col = config['target_col']
    out_dir = config['out_dir']
    if not os.path.isdir(out_dir):
        print(f"creating dir {out_dir}")
        os.mkdir(out_dir)

    locus2info, XY_df, loc2seq, conditions, cond_dict = load_data(config)
    print("Done Loading.")
    
    # create default train/test/val splits
    full_train_df,test_df = tu.quick_split(XY_df)
    train_df, val_df = tu.quick_split(full_train_df)

    # save the dfs to the outdir for future debugging
    train_df.to_csv(f'{out_dir}/train_df.tsv',sep='\t',index=False)
    val_df.to_csv(f'{out_dir}/val_df.tsv',sep='\t',index=False)
    test_df.to_csv(f'{out_dir}/test_df.tsv',sep='\t',index=False)

    # get the input sequence length
    seq_len = len(train_df[seq_col].values[0])

    # DECISION: single or multi task?

    # +----------------+
    # | SINGLE TASK WF |
    # +----------------+
    print(f"Running single task learning for {target_col}...")
    X, y = make_st_skorch_dfs(full_train_df, seq_col=seq_col,target_col=target_col)
    print("X:",X.shape)
    print("y:",y.shape)

    # make skorch object
    model_type = get_model_choice(config['model_type'])

    net_search = NeuralNetRegressor(
        model_type,
        module__seq_len=300,
        max_epochs=config['epochs'],
        #lr=0.001,
        device=DEVICE,#'cuda',  # uncomment this to train with CUDA
        verbose=1,
        callbacks=[
            Checkpoint(dirname=out_dir,f_pickle='best_chkpt.pkl'), # load_best=True
            EarlyStopping(patience=config['patience'])]
    )

    # make sklearn search object
    search = RandomizedSearchCV(
        net_search, 
        config['skorch_params'], 
        n_iter=config['search_iters'], 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, 
        cv=5,#cv, 
        random_state=1,
        verbose=2 #2
    )

    # learn stuff
    print("Fitting...")
    search.fit(X,y)

    # print stuff
    print(search.best_params_)
    print(search.best_estimator_)

    print("Saving searchCV obj....")
    # save search obj
    search_dump_file = open(f"{out_dir}/{config['job_name']}.pkl",'wb')
    pickle.dump(search, search_dump_file)

    # viz stuff

    # train/val loss plot
    search_label = [
        (
            (
            search.best_estimator_.history[:, 'train_loss'], 
            search.best_estimator_.history[:, 'valid_loss']
            ), 
        config['job_name'])
    ]
    loss_plot_filename = f"{out_dir}/{config['job_name']}_loss_plot.png"
    quick_loss_plot_simple(search_label,loss_plot_filename)

    # results
    res_df = pd.DataFrame(search.cv_results_)
    res_df['opt_name'] = res_df['param_optimizer'].apply(lambda x: x.__name__)
    res_df.to_csv(f"{out_dir}/{config['job_name']}_skres_df.tsv",sep='\t', index=False)

    # viz train/test
    Xtest, ytest = make_st_skorch_dfs(test_df, seq_col=seq_col,target_col=target_col)

    # y pred and pearson on training data
    ypred_train_search = search.best_estimator_.predict(X)
    p_train_search = scipy.stats.pearsonr(np.array(y).flatten(),ypred_train_search.flatten())[0]

    # y pred and pearson on test data
    ypred_test_search = search.best_estimator_.predict(Xtest)
    p_test_search = scipy.stats.pearsonr(np.array(ytest).flatten(),ypred_test_search.flatten())[0]

    pp_train_str = f"{config['job_name']}_train"
    pp_test_str = f"{config['job_name']}_test"
    parity_plot(pp_train_str, y, ypred_train_search,p_train_search,rigid=True,out_dir=out_dir)
    parity_plot(pp_test_str, ytest, ypred_test_search,p_test_search,rigid=True,out_dir=out_dir)

    print("DONE!")
    

if __name__ == '__main__':
    main()