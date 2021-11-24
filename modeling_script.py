# run a bunch of models and save them

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import pandas as pd
import random
random.seed(7)

import utils as u
import torch_utils as tu
from torch_utils import DatasetSpec

from models import DNA_Linear_Deep, Kmer_Linear, TINKER_DNA_CNN,DNA_LSTM,DNA_CNNLSTM



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_TYPES = [
    DatasetSpec('ohe'),
    DatasetSpec('kmer',k=3),
    DatasetSpec('kmer',k=6),
]

def setup_config():

    config = {
        'out_dir':'pipe1',
        #'model_types':['LinearDeep','CNN32','CNN128','Kmer3','Kmer6'],
        'model_types':['LSTM','CNNLSTM'],
        'learning_rates':[0.01,0.001],
        'sampler_types': ["default", "rebalanced"],
        'augmentation': [
            ("no",{}),
            ("revslide",{'stride':50}),
            ("mutation",{'mutation_rate':0.03}),
            ("mutation",{'mutation_rate':0.1}),
        ],

        'target_cond':'highCu',
        'seq_col':'upstream_region',
        'id_col':'locus_tag',
        'loss_func':nn.CrossEntropyLoss(),
        'loss_label':'Cross Entropy Loss',
        'epochs':5000
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

    # CNN 128 filt
    elif choice == "CNN128":
        cnn = TINKER_DNA_CNN(
            seq_len,
            num_filters0=128,
            num_filters1=32,
            kernel_size0=8,
            kernel_size1=8,
            conv_pool_size0=3,
            fc_node_num0=10,
            fc_node_num1=10
        )
        cnn.to(DEVICE)
        return cnn

    # LSTM
    elif choice == "LSTM":
        lstm = DNA_LSTM(
            seq_len,
            DEVICE,
            hidden_dim=100
        )
        lstm.to(DEVICE)
        return lstm

    # CNN-LSTM
    elif choice == "CNNLSTM":
        cnnlstm = DNA_CNNLSTM(
            seq_len,
            DEVICE,
            hidden_dim=100,
            num_filters=32,
            kernel_size=8
        )
        cnnlstm.to(DEVICE)
        return cnnlstm

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
        aug_str = 'no_aug'
        return train_df,aug_str

    # revcomp slide
    elif choice == 'revslide':
        temp = u.augment_revcomp(train_df)
        temp = u.augment_slide(temp,300,loc2flankseq,s=args['stride'])
        aug_str = f"revslide{args['stride']}"
        return temp,aug_str

    # mutation 
    elif choice == 'mutation':
        temp = u.augment_mutate(train_df,10,mutation_rate=args['mutation_rate'])
        aug_str = f"mutation{args['mutation_rate']}"
        return temp,aug_str

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
    target_cond = config['target_cond']
    seq_col = config['seq_col']
    id_col = config['id_col']
    loss_func = config['loss_func']
    loss_label = config['loss_label']
    epochs = config['epochs']
    out_dir = config['out_dir']

    if not os.path.isdir(out_dir):
        raise ValueError(f"{out_dir} does not exist. Please make it.")

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
    reg = tu.set_reg_class_up_down(XYdf,target_cond,thresh=0.6)

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

    # save the dfs to the outdir for future debugging
    train_df.to_csv(f'{out_dir}/train_df.tsv',sep='\t',index=False)
    val_df.to_csv(f'{out_dir}/val_df.tsv',sep='\t',index=False)
    test_df.to_csv(f'{out_dir}/test_df.tsv',sep='\t',index=False)

    # print("Train")
    # print(train_df[reg].value_counts())
    # print("Val")
    # print(val_df[reg].value_counts())
    # print("Test")
    # print(test_df[reg].value_counts())

    oracle = dict([(a,[b]) for a,b in XYdf[[id_col,reg]].values])
    seq_len = len(train_df[seq_col].values[0])

    # ** COLLECT RESULTS **
    res_rows = []
    loss_dict = {}
    
    # DATA AUGMENTATION LOOP
    for aug_choice,args in config['augmentation']:
        # augment the train df if needed
        aug_df,aug_str = get_augmentation_choice(aug_choice,args,train_df,loc2flankseq)
        print(f"Augmentation: {aug_str}")
        aug_df.to_csv(f'{out_dir}/aug_train_df.tsv',sep='\t',index=False)

        # sampler loop
        for sampler_choice in config['sampler_types']:
            print(f"\tSampler: {sampler_choice}")
            sampler, shuffle = get_sampler_choice(sampler_choice,aug_df,reg)

            # learning rate loop
            for lr in config['learning_rates']:
                print(f"\t\tLR: {lr}")
                dls = tu.build_dataloaders_single(
                    aug_df, 
                    val_df, 
                    DATASET_TYPES,
                    seq_col=seq_col,
                    target_col=reg,
                    sampler=sampler,
                    shuffle=shuffle
                )

                # *********************************************
                # Currently hardcoded to make these DataLoaders
                kmer6_train_dl,kmer6_val_dl = dls['kmer_6']
                kmer3_train_dl,kmer3_val_dl = dls['kmer_3']
                ohe_train_dl,ohe_val_dl = dls['ohe']
                # *********************************************

                # model type loop
                for model_choice in config['model_types']:
                    print(f"\t\t\tModel: {model_choice}")
                    # result dict for this specific model
                    res_dict = {}
                    model = get_model_choice(model_choice,seq_len)

                    if model_choice == "Kmer3":
                        train_dl, val_dl = dls['kmer_3']
                        ds = DatasetSpec('kmer',k=3)
                    elif model_choice == "Kmer6":
                        train_dl, val_dl = dls['kmer_6']
                        ds = DatasetSpec('kmer',k=6)
                    else: # catches anything not a kmer model with One-hot encoding
                        train_dl, val_dl = dls['ohe']
                        ds = DatasetSpec('ohe')

                    print("\t\t\t\tTraining...")
                    train_losses, val_losses,estop,best_val_loss = tu.run_model(
                        train_dl, 
                        val_dl, 
                        model,
                        loss_func,
                        DEVICE,
                        lr=lr,
                        epochs=epochs
                    )

                    data_label = [((train_losses,val_losses),model_choice,estop,best_val_loss)]


                    # collect model results
                    res_dict['train_losses'] = train_losses
                    res_dict['val_losses'] = val_losses
                    res_dict['estop'] = estop
                    res_dict['best_val_loss'] = best_val_loss
                    res_dict['data_label'] = data_label

                    # save model itself
                    print("\t\t\t\tSaving model...")
                    lr_str = f"_lr{lr}"
                    sample_str = f"_{sampler_choice}Sampler"
                    model_base_str = f"{model_choice}{lr_str}{sample_str}_{aug_str}"
                    model_filename = f"{model_base_str}.pth"
                    model_path = os.path.join(out_dir,model_filename)
                    torch.save(model,model_path)

                    # save loss data
                    res_dict_filename = f"{model_base_str}_loss_dict.npy"
                    res_dict_path = os.path.join(out_dir,res_dict_filename)
                    np.save(res_dict_path, res_dict) 

                    loss_dict[model_base_str] = res_dict

                    # get confusion data
                    print("\t\t\t\tGetting Confusion data...")
                    train_seqs = aug_df[id_col].values
                    train_conf_df = tu.get_confusion_data(model, model_choice, ds, train_seqs, oracle,loc2seq,DEVICE)
                    train_conf_df.to_csv(f"{model_base_str}_train_conf_df.tsv",sep='\t',index=False)

                    val_seqs = val_df[id_col].values
                    val_conf_df = tu.get_confusion_data(model, model_choice, ds, val_seqs, oracle,loc2seq,DEVICE)
                    val_conf_df.to_csv(f"{model_base_str}_val_conf_df.tsv",sep='\t',index=False)

                    # get classification report
                    cls_report = tu.cls_report(val_conf_df)

                    # put into result row
                    row = [
                        model_base_str,
                        model_choice,
                        lr,
                        sampler_choice,
                        aug_str,
                        estop,
                        best_val_loss,
                        cls_report['acc'],
                        cls_report['mcc'],
                        cls_report['mi_p'],
                        cls_report['mi_r'],
                        cls_report['mi_f1'],
                        cls_report['ma_p'],
                        cls_report['ma_r'],
                        cls_report['ma_f1']
                    ]
                    res_rows.append(row)

                    loss_dict[model_base_str] = res_dict

                    print(f"\t\t\t\tDone with {model_choice}...")
    cols = [
        'model_desc',
        'model_type',
        'lr',
        'sampler',
        'data_aug',
        'epoch_stop',
        'best_val_loss',
        'acc',
        'mcc',
        'mi_p',
        'mi_r',
        'mi_f1',
        'ma_p',
        'ma_r',
        'ma_f1'
    ]
    res_df = pd.DataFrame(res_rows,columns=cols)
    res_path = os.path.join(out_dir,'res_df.tsv')
    res_df.to_csv(res_path,sep='\t',index=False)

    loss_dict_path = os.path.join(out_dir,'loss_dict.npy')
    np.save(loss_dict_path, loss_dict) 
    print("Done")


    

if __name__ == '__main__':
    main()
