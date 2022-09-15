# attempt to recreate slurm-able skorch script
# for a multitask learning setting
# adapting for E. coli iModulon prediction

import argparse
import torch
from torch import nn
from torch.nn import functional as F

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import seaborn as sns
import scipy
import yaml

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform,uniform
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import multilabel_confusion_matrix,ConfusionMatrixDisplay,classification_report

from skorch import NeuralNetRegressor,NeuralNetClassifier
from skorch.callbacks import EarlyStopping,Checkpoint,GradientNormClipping
from skorch.dataset import Dataset
from skorch.helper import predefined_split


# print("PRE-DASK")
# # atempted DASK stuff?
# from dask.distributed import Client
# from joblib import parallel_backend
# print("Post-DASK import")
# client = Client('127.0.0.1:8786')
# print("Post-DASK client start")
# client.upload_file("models.py")
# # client.upload_file("utils.py")
# # client.upload_file("torch_utils.py")
# print("Post-DASK client upload")

import utils as u 
import torch_utils as tu
import models as m 


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
set_seed(46)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:",DEVICE)

def load_iMod_binarized_data(config):
    # +-----------------+
    # | Load iMod files |
    # +-----------------+
    
    # load file with iModulon data (gene info + M values)
    XY = pd.read_csv(config['imod_info_m'],sep='\t')
    
    id_col = config['id_col']
    seq_col = config['seq_col']
    loc2seq = dict([(x,z) for (x,z) in XY[['locus_tag','upstream_region']].values])

    # get iModulon column labels
    M = pd.read_csv(config['M_matrix'],index_col=0)
    imods = [x.strip() for x in M.columns]

    # load binarized version of M matrix
    Mb = pd.read_csv(config['gene_presence_matrix'],index_col=0).astype(int)
    Mb.index.name='locus_tag'

    # Convert XY into binarazed version
    XYb = pd.merge(XY.drop(imods,axis=1), Mb.reset_index(),on='locus_tag')
    XYb
        
    # +------------------------+
    # | Filter out small iMods |
    # +------------------------+
    thresh = config['imod_count_thresh']
    XYim = XYb[imods]
    # m binarized count dict
    mbc = dict([(XYim.T.iloc[i].name,sum(XYim.T.iloc[i])) for i in range(XYim.shape[1])])
    imods_filt = [x for x in mbc if mbc[x]>thresh]
    print(f"num iMods that pass threshold of {thresh}:",len(imods_filt))

    # these iMods seem too small to meaningfully try to predict
    # (some are single-gene iMods)
    imods_exclude = [x for x in mbc if mbc[x]<=thresh]
    print(f"num iMods filtered out b/c fewer than {thresh+1} genes: {len(imods_exclude)}")

    # drop excluded iMods and keep binary labels
    XY = XYb.drop(imods_exclude,axis=1)
    XY

    return XY, loc2seq, imods, imods_filt, 

def get_opts(opts):
    '''
    Given a list of optimizer names as strings, return the torch.optim object.
    (Is there a way to specify the optimizer via string in PyTorch?)
    '''
    optimizers = {
        'Adam': torch.optim.Adam,
        'AdamW' : torch.optim.AdamW,
        'Adagrad' : torch.optim.Adagrad,
        'RMSprop' : torch.optim.RMSprop,
        'SGD' : torch.optim.SGD,
    }
    for opt in opts:
        if opt not in optimizers:
            raise ValueError(f"{opt} optimizer not in list of options. Currently available: {optimizers.keys()}")
    else:
        return [optimizers[opt] for opt in opts]


def get_params(filename):

    with open(filename) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['optimizer'] = get_opts(params['optimizer'])
    
    return params 


def setup_config(filename):

    with open(filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['skorch_params'] = get_params(config['param_file']),

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
    
    # LINEAR
    if choice == 'LinearDeep':
        return m.DNA_Linear_Deep

    # CNN 
    elif choice == "CNN":
        return m.DNA_CNN
    elif choice == "2CNN":
        return m.DNA_2CNN

    # CNN Multi
    elif choice == "2CNN_Multi":
        return m.DNA_2CNN_Multi

    # LSTM
    elif choice == "LSTM":
        return m.DNA_LSTM

    # CNN-LSTM
    elif choice == "CNNLSTM":
        return m.DNA_CNNLSTM

    # Kmer 
    elif choice == "Kmer":
        return m.Kmer_Linear

    # 2CNN_2FC
    elif choice == 'DNA_2CNN_2FC_Multi':
        return m.DNA_2CNN_2FC_Multi


    else:
        choices = ['LinearDeep', 'CNN', '2CNN', 'LSTM','CNNLSTM','Kmer',
               '2CNN_Multi','DNA_2CNN_2FC_Multi'
        ]
        raise ValueError(f"{choice} model choice not recognized. Options are: {choices}")


def get_class_counts(ys):
    '''
    Given a list of iMod vector labels, sum the number of
    positive examples for each iModulon
    '''
    y_sum = torch.tensor(np.zeros(ys.shape[1]))
    for ex in ys:
        y_sum += ex

    return y_sum

# https://discuss.pytorch.org/t/weights-in-bcewithlogitsloss/27452/11?u=crypdick
def get_pos_weights(ys):
    '''
    Determine loss reweighting vector by the inverse of the positive
    examples for each iMod task
    '''
    class_counts = get_class_counts(ys)
    pos_weights = np.ones_like(class_counts)
    neg_counts = [len(ys)-pos_count for pos_count in class_counts]  # <-- HERE 
    
    for cdx, (pos_count, neg_count) in enumerate(zip(class_counts,  neg_counts)):
        #print(f"{cdx}| pos:{pos_count}  neg:{neg_count}")
        #print("val:", neg_count / (pos_count + 1e-5))
        pos_weights[cdx] = neg_count / (pos_count + 1e-5)
        

    return torch.as_tensor(pos_weights, dtype=torch.float)


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

def alt_cls_summary2(df):
    heat = alt.Chart(df).mark_rect().encode(
        x=alt.X('imod:O'),
        y=alt.Y('metric:O'),
        color=alt.Color('score:Q',scale=alt.Scale(domain=(0.0,1.0))),
        tooltip=['metric:N','score:Q']
    ).properties(width=600)

    support = alt.Chart(df).mark_bar().encode(
        x=alt.X('imod:O',title='',axis=alt.Axis(labels=False)),
        color=alt.Color('support:Q', legend=None,scale=alt.Scale(scheme='greys')),
        y='support:Q',
        tooltip=['support']
    ).properties(width=600,height=50)

    return alt.vconcat(support,heat
        ).resolve_scale(color='independent'
        ).configure_concat(
            spacing=0
        )

def view_cls_report(sk_model,Xs,ys,imods):
    '''
    For a given model and set of X,y examples, save and display 
    a summary of the primary classification metrics
    '''
    # get the predictions and classification report
    y_preds = sk_model.predict(Xs)
    cls_rep = classification_report(ys, y_preds,target_names=imods,output_dict=True)
    
    # conver the dict into a df for viewing
    cls_df = pd.DataFrame.from_dict(cls_rep,orient='index')
    cls_df.index.name='imod'
    cls_df = cls_df.reset_index()
    
    # drop the micro/macro average colums
    cls_df = cls_df.drop(cls_df[~cls_df['imod'].isin(imods)].index)
    # convert to int for sorting
    cls_df['imod'] = cls_df['imod'].apply(lambda x: int(x))
    
    # melt the df for altair
    cls_melt = cls_df.melt(
        id_vars=['imod','support'],
        value_vars=['precision','recall','f1-score'],
        var_name='metric',
        value_name='score')
    
    #alt_cls_summary(cls_melt)
    chart = alt_cls_summary2(cls_melt)

    return cls_df,chart

# #####################################################
def main():
    parser = argparse.ArgumentParser(description='Run hyperparam search with skorch and sklearn.')
    
    # Required args
    parser.add_argument('config_file', help='config file specified as yaml')

    args = parser.parse_args()

    # +----------------------+
    # | Load data and config |
    # +----------------------+
    config = setup_config(args.config_file)
    id_col = config['id_col']
    seq_col = config['seq_col']

    out_dir = config['out_dir']
    if not os.path.isdir(out_dir):
        print(f"creating dir {out_dir}")
        os.mkdir(out_dir)

    XY, loc2seq, imods_all, target_cols = load_iMod_binarized_data(config)
    # imods_filt are target_cols

    print("Done Loading.")
    print("Full df shape:", XY.shape)

    # +------------------------+
    # | Multi-task training WF |
    # +------------------------+
    print(f"Running MULTI task learning for {target_cols}...")
    X, y = make_mt_skorch_dfs(XY, seq_col=seq_col,target_cols=target_cols)
    print("X:",X.shape)
    print("y:",y.shape)

    # use stratified splitting across the tasks
    # (best effort to ensure the train/test/val splits
    # each have some positive examples of each task)
    Xfull_train_strat, yfull_train_strat, Xtest_strat, ytest_strat = iterative_train_test_split(X, y, test_size = 0.2)
    Xtrain_strat, ytrain_strat, Xval_strat, yval_strat = iterative_train_test_split(Xfull_train_strat, yfull_train_strat, test_size = 0.2)
    valid_ds = Dataset(Xval_strat, yval_strat)

    # get weights for binary cross entropy loss
    bce_pos_weights = get_pos_weights(y)

    # get the input sequence length
    seq_len = len(XY[seq_col].values[0])

    model_type = get_model_choice(config['model_type'])
    
    # make skorch object
    net_search = NeuralNetClassifier(
        model_type,
        module__seq_len=seq_len,
        module__n_tasks=y.shape[1],
        max_epochs=config['epochs'],
        #lr=0.001,
        device=DEVICE,#'cuda',  # uncomment this to train with CUDA
        verbose=0, # without 0 it prints the every epoch loss
        callbacks=[
            Checkpoint(dirname=out_dir,f_pickle='best_chkpt.pkl'), # load_best=True
            EarlyStopping(patience=config['patience']),
            GradientNormClipping(),
            ],
        train_split=predefined_split(valid_ds),
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=bce_pos_weights)
    )

    # make sklearn search object
    search = RandomizedSearchCV(
        net_search, 
        config['skorch_params'], 
        n_iter=config['search_iters'], 
        scoring=['f1_macro'], 
        refit='f1_macro',
        n_jobs=-1, # TODO: set this more explicitly
        cv=3,#cv, 
        random_state=1,
        verbose=1
    )

    # learn stuff
    #with parallel_backend('dask'):
    print("Fitting...")
    search.fit(Xtrain_strat,ytrain_strat)

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
    quick_loss_plot_simple(search_label,loss_plot_filename,loss_type='BCEWithLogitsLoss')

    # results
    res_df = pd.DataFrame(search.cv_results_)
    res_df['opt_name'] = res_df['param_optimizer'].apply(lambda x: x.__name__)
    res_df.to_csv(f"{out_dir}/{config['job_name']}_skres_df.tsv",sep='\t', index=False)

    # Create classification reports for full train and test
    cls_full_train_df,full_train_metric_chart = view_cls_report(search.best_estimator_,Xtrain_strat,ytrain_strat,target_cols)
    full_train_metric_chart.save(f"{out_dir}/best_est_full_train_metric_chart.html")
    
    cls_test_df,test_metric_chart = view_cls_report(search.best_estimator_,Xtest_strat,ytest_strat,target_cols)
    test_metric_chart.save(f"{out_dir}/best_est_test_metric_chart.html")


    print("DONE!")
    

if __name__ == '__main__':
    main()