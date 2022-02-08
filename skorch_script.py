import torch
from torch import nn
from torch.nn import functional as F

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy

from skorch import NeuralNetRegressor

import utils as u 
import torch_utils as tu

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE


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

	return locus2info, XY_df, loc2seq


def get_params():
	# params specific for CNN of a certain type...
	params = {
	    'lr': [0.0005, 0.0001,0.00001],#loguniform(0.0001, 0.01)
	    
	    'module__num_filters': [16,32,64,128], # uniform(8,128), #
	    'module__kernel_size': [4,8,16,32],
	    'module__num_fc_nodes1': [10, 25, 50,100], #randint(10,100), #
	    'optimizer':[torch.optim.SGD, torch.optim.Adam, torch.optim.Adagrad,torch.optim.AdamW,torch.optim.RMSprop]
	    #'optimizer__nesterov': [False, True],
	}

	return params 

def setup_config():

    config = {
        'expression_file':'data/XY_logTPM_opFilt.tsv',
        'locus_info_file':'data/locus2info.tsv',
        'id_col':'locus_tag',
        'seq_col':'upstream_region',
        'out_dir':'skorch_test',
        'skorch_params':get_params(),
        'epochs':5000,

        
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


def make_mt_skorch_dfs(df,seq_col='seq',target_cols=conditions):
    
    seqs = list(df[seq_col].values)        
    ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in seqs])

    labels = torch.tensor(list(df[target_cols].values))
    # bad dimension? fixed in model.forward for now
    
    return ohe_seqs.float(), labels.float()


def main():
	config = setup_config()
    
    locus2info, XY_df, loc2seq = load_data()
    
    out_dir = config['out_dir']

    if not os.path.isdir(out_dir):
        print(f"creating dir {out_dir}")
        os.mkdir(out_dir)

    # create default train/test/val splits
    full_train_df,test_df = tu.quick_split(XY_df)
	train_df, val_df = tu.quick_split(full_train_df)


    # save the dfs to the outdir for future debugging
    train_df.to_csv(f'{out_dir}/train_df.tsv',sep='\t',index=False)
    val_df.to_csv(f'{out_dir}/val_df.tsv',sep='\t',index=False)
    test_df.to_csv(f'{out_dir}/test_df.tsv',sep='\t',index=False)


    seq_len = len(train_df[seq_col].values[0])

    


    

if __name__ == '__main__':
    main()