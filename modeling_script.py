# run a bunch of models and save them

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import random
random.seed(7)
import tqdm

import utils as u
import torch_utils as tu
from torch_utils import DatasetSpec

from models import DNA_Linear_Deep, Kmer_Linear, TINKER_DNA_CNN



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_config():

    config = {
        'model_types':['LinearDeep','CNN32','Kmer3','Kmer6'],
        'learning_rates':[0.01,0.001],
        'sampler_types': ["default", "rebalanced"],
        'augmentation': [
            ("no",{}),
            ("revslide",{'stride':50}),
            ("mutation",{'mutation_rate':0.03}),
            ("mutation",{'mutation_rate':0.1}),
        ],

        'reg':'highCu_reg_UD',
        'seq_col':'upstream_region',
        'id_col':'locus_tag'
    }

    return config

def get_model_choice(choice,seq_len):
    # LINEAR
    if choice == 'LinearDeep':
        lin_d = DNA_Linear_Deep(
            seq_len,
            h0_size=100,
            h1_size=100,
        )
        lin_d.to(DEVICE)
        return lin_d

    # CNN 32 filt
    elif choice == "CNN32":
        cnn = TINKER_DNA_CNN(
            seq_len,
            num_filters0=32,
            num_filters1=32,
            kernel_size0=8,
            kernel_size1=8,
            conv_pool_size0=3,
            fc_node_num0=10,
            fc_node_num1=10
        )
        cnn.to(DEVICE)
        return cnn

    # Kmer 3
    elif choice == "Kmer3":
        kmer = Kmer_Linear(
            64, # 4^3
            h1_size=100,
            h2_size=10,
        )
        kmer.to(DEVICE)
        return kmer

    # Kmer 6
    elif choice == "Kmer6":
        kmer = Kmer_Linear(
            4096, # 4^6
            h1_size=1000,
            h2_size=10,
        )
        kmer.to(DEVICE)
        return kmer

    else:
        raise ValueError(f"{choice} model choice not recognized. Options are: LinearDeep, CNN32, Kmer3, Kmer6")

def get_augmentation_choice(choice,args,train_df,loc2flankseq):
    # No Augmentation
    if choice == 'no':
        return train_df

    # revcomp slide
    elif choice == 'revslide':
        temp = u.augment_revcomp(train_df)
        temp = u.augment_slide(temp,300,loc2flankseq,s=args['stride'])
        return temp

    # mutation 
    elif choice == 'mutation':
        temp = u.augment_mutate(train_df,10,mutation_rate=args['mutation_rate'])
        return temp

    else:
        raise ValueError(f"{choice} data augmentation choice not recognized. Options are: no, revslide, mutation")

def get_sampler_choice(choice,train_df,reg):
    # Default
    if choice == 'default':
        return None,True

    # rebalanced
    elif choice == 'rebalanced':
        sampler = tu.make_weighted_sampler(train_df,reg)
        return sampler, False

    else:
        raise ValueError(f"{choice} sampler choice not recognized. Options are: default, rebalanced")


def filter_inactive_genes(df, tpm_thresh):
    # list of relevant condition names
    with open("data/conditions_to_include.txt",'r') as f:
        conds = [x.strip() for x in f.readlines()]

    # df with actual TPM counts
    data_filename = "data/XY_TPM_opFilt.tsv"
    tpm_df = pd.read_csv(data_filename,sep='\t')

    # collect genes that never express above a given threshold
    no_tx_genes = []
    for i, row, in tpm_df.iterrows():
        tpms = row[conds].values
        if max(tpms) < tpm_thresh:
            no_tx_genes.append(row['locus_tag'])
    
    # return only genes not in "no transcription genes"
    return df[~df['locus_tag'].isin(no_tx_genes)].reset_index().drop('index',axis=1)
    

def main():

    config = setup_config()
    reg = config['reg']
    seq_col = config['seq_col']
    id_col = config['id_col']

    # locus to gene info
    locus_info_filename = 'data/locus2info.tsv'
    locus_info_df = pd.read_csv(locus_info_filename,sep='\t')
    locus2info = u.make_info_dict(locus_info_df)

    # log ratio data file
    data_filename = "data/XY_lr_noCu_opFilt.tsv"
    XYdf_og = pd.read_csv(data_filename,sep='\t')
    loc2seq = dict([(x,z) for (x,z) in XYdf_og[[id_col,seq_col]].values])

    # file with promoter regions with extra flanking sequence
    flank_data_filename = "data/XY_lr_noCu_opFilt_-400:100.tsv"
    XYdf_flank = pd.read_csv(flank_data_filename,sep='\t')
    loc2flankseq = dict([(x,z) for (x,z) in XYdf_flank[[id_col,seq_col]].values])

    # filter out genes that never express above 2 tpm
    XYdf = filter_inactive_genes(XYdf_og,2)

    # set regulatory class
    tu.set_reg_class_up_down(XYdf,'highCu',thresh=0.6)

    # get stratified train/test/val split
    # specs for class partition dict
    cpd = {
        0: {'train_test':0.8, 'train_val':0.8},
        1: {'train_test':0.8, 'train_val':0.8},
        2: {'train_test':0.8, 'train_val':0.8},
    }

    full_train_df, \
    test_df, \
    train_df, \
    val_df = tu.stratified_partition(XYdf, cpd, class_col=reg)

    print("Train")
    print(train_df[reg].value_counts())
    print("Val")
    print(val_df[reg].value_counts())
    print("Test")
    print(test_df[reg].value_counts())

    seq_len = len(train_df[seq_col].values[0])
    
    # DATA AUGMENTATION LOOP
    for aug_choice,args in config['augmentation']:
        # augment the train df if needed
        aug_df = get_augmentation_choice(aug_choice,args,train_df,loc2flankseq)

        # sampler loop
        for sampler_choice in config['sampler_types']:
            sampler, shuffle = get_sampler_choice(sampler_choice,aug_df,reg)

            # learning rate loop
            for lr in config['learning_rates']:
                # MAKE DLS HERE

                # model type loop
                for model_choice in config['model_types']:
                    model = get_model_choice(model_choice,seq_len)

                    print(f"{aug_choice} | {lr} | {sampler_choice} | {model_choice}")

    

if __name__ == '__main__':
    main()
