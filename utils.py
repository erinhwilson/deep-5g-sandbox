import numpy as np
import pandas as pd

from override import GENE_NAME_OVERRIDE, GENE_PRODUCT_OVERRIDE, SYS_LOOKUP

def load_promoter_seqs(filename):
    '''
    Load fasta file of promoters into ID, desc, and seq. It expects
    each fasta header to be divided by "|" with in the format:
    LOCUS_TAG|GENE_SYMBOL|PRODUCT
    '''
    proms = []
    with open(filename,'r') as f:
        for line in f:
            if line.startswith(">"):
                full_header = line.strip()[1:].strip()
                locus_tag = full_header.split('|')[0]
            else:
                seq = line.strip().upper()
                proms.append((locus_tag,full_header,seq))
                
    return proms

def make_info_dict(df):
    '''
    Given a TPM df, make a dict with some metadata keyed by
    locus tag
    '''
    
    def get_info_for_row(row):
        d = {
        'gene':row[1],
        'product':row[2],
        'type':row[3],
        }
        return d

        
    info = df[['locus_tag','gene_symbol','product','type']].values
    locus2info = dict(
        [(row[0],get_info_for_row(row)) for row in info]
    )

    # update with overrides
    for loc in GENE_NAME_OVERRIDE:
        locus2info[loc]['gene'] = GENE_NAME_OVERRIDE[loc]
    for loc in GENE_PRODUCT_OVERRIDE:
        locus2info[loc]['product'] = GENE_PRODUCT_OVERRIDE[loc]
        
    return locus2info
    
    
    
def get_gene_means_by_condition(df,samples,sample2condition):
    '''
    Given a df of genes with samples in the columns, return a df with
    conditions as the rows and columns that are genes with values averaged
    by condition
    '''
    # transpose dataframe to make experiments the rows and genes the columns
    df_T = df.set_index('locus_tag')[samples].T.reset_index().rename(columns={'index':'sample'})
    # ^^ kinda gross pandas syntax... try to understand what each part does
    df_T['exp_condition'] = df_T['sample'].apply(lambda x: sample2condition[x])
    
    # list of all locus tags
    loci = list(df['locus_tag'].values)

    # get average value of each locus_tag in each condition
    # (unique same ids go away)
    df_means = df_T[['exp_condition']+loci]\
                        .groupby('exp_condition',as_index=False)\
                        .mean()

    return df_means
    
def one_hot_encode(seq):
    #print("one hot encoding...")
    
    # Dictionary returning one-hot encoding of nucleotides. 
    nuc_d = {'A':[1.0,0.0,0.0,0.0],
             'C':[0.0,1.0,0.0,0.0],
             'G':[0.0,0.0,1.0,0.0],
             'T':[0.0,0.0,0.0,1.0],
             'N':[0.0,0.0,0.0,0.0]}
    
    # Creat empty matrix.
    #vec=torch.tensor([nuc_d[x] for x in seq])
    vec=np.array([nuc_d[x] for x in seq]).flatten()
        
    return vec
    
