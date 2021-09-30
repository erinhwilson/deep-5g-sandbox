# torch_utils.py
# functions for doing things in Pytorch
import altair as alt
import logomaker
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import random
random.seed(7)

import torch
from torch.utils.data import Dataset,DataLoader #,TensorDataset
from torch import nn

import utils as u

# +------------------------+
# | Custom Dataset classes |
# +------------------------+
class SeqDatasetOHE(Dataset):
    '''
    Multi-task for one-hot-encoded sequences
    '''
    def __init__(self,df,seq_col='seq',target_col='score'):
        self.seqs = list(df[seq_col].values)
        self.seq_len = len(self.seqs[0])
        
        self.ohe_seqs = torch.stack([torch.tensor(u.one_hot_encode(x)) for x in self.seqs])
    
        self.labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)
        
    def __len__(self): return len(self.seqs)
    
    def __getitem__(self,idx):
        seq = self.ohe_seqs[idx]
        label = self.labels[idx]
        
        return seq, label
    
class SeqDatasetKmer(Dataset):
    '''
    Multi-task for k-mer vector sequences
    '''
    def __init__(self,df,k=3,seq_col='seq',target_col='score'):
        self.seqs = list(df[seq_col].values)
        self.seq_len = len(self.seqs[0])
        self.kmers = u.kmers(k)
        
        self.kmer_vecs = torch.stack([torch.tensor(u.count_kmers_in_seq(x,self.kmers)) for x in self.seqs])
    
        self.labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)
        
    def __len__(self): return len(self.seqs)
    
    def __getitem__(self,idx):
        kmer_vec = self.kmer_vecs[idx]
        label = self.labels[idx]
        
        return kmer_vec, label


class DatasetSpec():
    '''
    Quick access class for speciying the type of dataset to build
    '''
    def __init__(self,ds_type,k=None):
        self.name = ds_type
        self.k = k
        
        self.id = self.name if not k else f"{self.name}_{k}"

# +-----------------------+
# | Dataloaders functions |
# +-----------------------+

def quick_split(df, split_frac=0.8, verbose=False):
    '''
    Given a df of samples, randomly split indices between
    train and test at the desired fraction
    '''
    df = df.reset_index()

    # shuffle indices
    idxs = list(range(df.shape[0]))
    random.shuffle(idxs)

    # split shuffled index list by split_frac
    split = int(len(idxs)*split_frac)
    train_idxs = idxs[:split]
    test_idxs = idxs[split:]
    
    # split dfs and return
    train_df = df[df.index.isin(train_idxs)]
    test_df = df[df.index.isin(test_idxs)]
        
    return train_df, test_df

def build_dataloaders_single(train_df,
                             test_df, 
                             ds_specs,
                             seq_col='seq',
                             target_col="score",
                             batch_size=128,
                             split_frac=0.8
                            ):
    '''
    Given a df, split into train and test, and encode the sequence for modeling 
    based on the requested dataset types (eg OHE or Kmer counts). Load each 
    Dataset into a pytorch loaders. 
    '''
    
    # split
    #train_df, test_df = u.quick_split(df,split_frac=split_frac)
    
    dls = {} # collect data loaders
    
    for ds in ds_specs:
        # Kmer data set
        if ds.name == 'kmer':
            if not ds.k:
                raise ValueError(f"To use SeqDatasetKmer, you must specify an integer value for k in DatasetSpec")
            assert(type(ds.k) == int)
            
            train_ds = SeqDatasetKmer(train_df, ds.k,seq_col=seq_col,target_col=target_col)
            test_ds = SeqDatasetKmer(test_df, ds.k,seq_col=seq_col,target_col=target_col)
            
        # One-hot encoding
        elif ds.name == 'ohe':
            train_ds = SeqDatasetOHE(train_df,seq_col=seq_col,target_col=target_col)
            test_ds = SeqDatasetOHE(test_df,seq_col=seq_col,target_col=target_col)
            
        # unknown datatype?
        else:
            raise ValueError(f"Unknown Dataset Type {ds.name}.")

        # Put DataSets into DataLoaders
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)
        dls[ds.id] = (train_dl,test_dl)
    
    return dls


# +--------------------------------+
# | Training and fitting functions |
# +--------------------------------+

def loss_batch(model, loss_func, xb, yb, opt=None,verbose=False):
    '''
    Apply loss function to a batch of inputs. If no optimizer
    is provided, skip the back prop step.
    '''
    if verbose:
        print('loss batch ****')
        print("xb shape:",xb.shape)
        print("yb shape:",yb.shape)

    xb_out = model(xb.float())
    if verbose:
        print("model out pre loss", xb_out.shape)
    loss = loss_func(xb_out, yb.float())

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    #print("lb returning:",loss.item(), len(xb))
    return loss.item(), len(xb)


def train_step(model, train_dl, loss_func, device, opt):
    '''
    Execute 1 set of batched training within an epoch
    '''
    # Set model to Training mode
    model.train()
    tl = [] # train losses
    ns = [] # batch sizes, n
    # collect train loss; provide opt so backpropo happens
    for xb, yb in train_dl:
        # put on GPU
        xb, yb = xb.to(device),yb.to(device)

        t, n = loss_batch(model, loss_func, xb, yb, opt=opt)
        tl.append(t)
        ns.append(n)
    
    # average the losses over all batches    
    train_loss = np.sum(np.multiply(tl, ns)) / np.sum(ns)
    
    return train_loss

def test_step(model, test_dl, loss_func, device):
    '''
    Execute 1 set of batched validation within an epoch
    '''
    # Set model to Evaluation mode
    model.eval()
    with torch.no_grad():
        tl = [] # test losses
        ns = [] # batch sizes
        for xb, yb in test_dl:
            # put on GPU
            xb, yb = xb.to(device),yb.to(device)

            t, n = loss_batch(model, loss_func, xb, yb)
            tl.append(t)
            ns.append(n)

    # average the losses over all batches
    test_loss = np.sum(np.multiply(tl, ns)) / np.sum(ns)
    
    return test_loss


def fit(epochs, model, loss_func, opt, train_dl, test_dl, device):
    '''
    Fit the model params to the training data, eval on unseen data.
    Loop for a number of epochs and keep train of train and test losses 
    along the way
    '''
    # keep track of losses
    train_losses = []    
    test_losses = []
    
    # loops through epochs
    for epoch in range(epochs):
        # train step
        train_loss = train_step(model, train_dl, loss_func, device, opt)
        train_losses.append(train_loss)
        
        # test step
        test_loss = test_step(model, test_dl, loss_func, device)
        print(epoch, test_loss)
        test_losses.append(test_loss)

    return train_losses, test_losses


def run_model(train_dl,test_dl, model, device,lr=0.01, epochs=20):
    '''
    Given data and a model type, run dataloaders with MSE loss and SGD opt
    '''
    # define loss func and optimizer
    loss_func = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
    
    # run the training loop
    train_losses, test_losses = fit(epochs, model, loss_func, optimizer, train_dl, test_dl, device)
    
    #return model, train_losses, test_losses
    return train_losses, test_losses


# +--------------------------------+
# | Prediction checking/validation |
# +--------------------------------+

def quick_seq_pred(model, seqs, oracle):
    '''
    Given a model and some sequences, get the model's predictions
    for those sequences and compare to the oracle (true) output
    '''
    for dna in seqs:
        s = torch.tensor(u.one_hot_encode(dna)).unsqueeze(0)
        pred = model(s.float())
        actual = oracle[dna]
        diff = actual - pred.item()
        print(f"{dna}: pred:{pred.item():.3f} actual:{actual:.3f} ({diff:.3f})")


def parity_plot(model_name,df, pearson):
    '''
    Given a dataframe of samples with their true and predicted values,
    make a scatterplot.
    '''
    plt.scatter(df['truth'].values, df['pred'].values, alpha=0.2)
    
    # y=x line
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False, scaley=False)

    plt.ylabel("Predicted Score",fontsize=14)
    plt.xlabel("Actual Score",fontsize=14)
    plt.title(f"{model_name} (pearson:{pearson:.3f})",fontsize=20)
    plt.show()
    
def alt_parity_plot(model,df, pearson,task):
    '''
    Make an interactive parity plot with altair
    '''
    chart = alt.Chart(df).mark_circle(opacity=0.2).encode(
        alt.X('pred:Q'),
        alt.Y('truth:Q'),
        tooltip=['seq:N']
    ).properties(
        title=f'Model (pearson:{pearson})'
    ).interactive()
    
    chart.save(f'alt_out/parity_plot_{task}_{model}.html')
    

def parity_pred(models, seqs, oracle,task,alt=True):
    '''Given some sequences, get the model's predictions '''
    dfs = {} # key: model name, value: parity_df
    
    
    for model_name,model in models:
        print(f"Running {model_name}")
        data = []
        for dna in seqs:
            s = torch.tensor(u.one_hot_encode(dna))#.unsqueeze(0)
            actual = oracle[dna]
            pred = model(s.float())
            data.append([dna,actual,pred.item()])
        df = pd.DataFrame(data, columns=['seq','truth','pred'])
        pearson = df['truth'].corr(df['pred'])
        dfs[model_name] = (pearson,df)
        
        #plot parity plot
        if alt: # make an altair plot
            alt_parity_plot(model_name, df, pearson,task)
        parity_plot(model_name, df, pearson)

    return dfs

def quick_loss_plot(data_label_list):
    '''
    For each train/test loss trajectory, plot loss by epoch
    '''
    for (train_data,test_data),label in data_label_list:
        plt.plot(train_data,linestyle='--',color=f"C{i}", label=f"{label} Train")
        plt.plot(test_data,color=f"C{i}", label=f"{label} Test")

    plt.legend()
    plt.ylabel("MSE loss")
    plt.xlabel("Epoch")
    plt.show()


# +------------------------------+
# | CNN filter viewing functions |
# +------------------------------+

def get_conv_layers_from_model(model):
    '''
    Given a trained model, extract its convolutional layers
    '''
    model_children = list(model.children())
    
    # counter to keep count of the conv layers
    model_weights = [] # we will save the conv layer weights in this list
    conv_layers = [] # we will save the actual conv layers in this list
    bias_weights = []
    counter = 0 

    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        # get model type of Conv1d
        if type(model_children[i]) == nn.Conv1d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
            bias_weights.append(model_children[i].bias)

        # also check sequential objects' children for conv1d
        elif type(model_children[i]) == nn.Sequential:
            for child in model_children[i]:
                if type(child) == nn.Conv1d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
                    bias_weights.append(child.bias)

    print(f"Total convolutional layers: {counter}")
    return conv_layers, model_weights, bias_weights

def view_filters(model_weights, num_cols=8):
    num_filt = model_weights[0].shape[0]
    filt_width = model_weights[0][0].shape[1]
    num_rows = int(np.ceil(num_filt/num_cols))
    
    # visualize the first conv layer filters
    plt.figure(figsize=(20, 17))

    for i, filter in enumerate(model_weights[0]):
        ax = plt.subplot(num_rows, num_cols, i+1)
        ax.imshow(filter.detach(), cmap='gray')
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(['A', 'C', 'G','T'])
        ax.set_xticks(np.arange(filt_width))
        ax.set_title(f"Filter {i}")

    plt.tight_layout()
    plt.show()
    
def get_conv_output_for_seq(seq, conv_layer):
    '''
    Given an input sequeunce, get the output tensor containing the filter activations
    '''
    print(f"Running seq {seq}")
    # format seq for input to conv layer (OHE, reshape)
    seq = torch.tensor(u.one_hot_encode(seq))#.view(-1,len(seq),4).permute(0,2,1)
    # OHE FIX??
    
    # run through conv layer
    with torch.no_grad(): # don't want as part of gradient graph?
        res = conv_layer(seq.float())
        return res[0]
    

def get_filter_activations(seqs, conv_layer):
    '''
    Given a set of input sequences and a trained convolutional layer, 
    determine the subsequences for which each filter in the conv layer 
    activate most strongly. 
    
    1.) Run inputs through conv layer. 
    2.) Loop through filter activations of the resulting tensor, saving the
            position where filter activations were >0. 
    3.) Compile a count matrix for each filter by accumulating subsequences which
            activate the filter
    '''
    # initialize dict of pwms for each filter in the conv layer
    num_filters = conv_layer.out_channels
    filt_width = conv_layer.kernel_size[0]
    filter_pwms = dict((i,torch.zeros(4,filt_width)) for i in range(num_filters))
    
    # loop through a set of sequences and collect subseqs where each filter activated
    for seq in seqs:
        res = get_conv_output_for_seq(seq, conv_layer)
        # for each filter and it's activation vector
        for filt_id,act_vec in enumerate(res):
            activated_positions = [x.item() for x in torch.where(act_vec>0)[0]]
            
            # get subsequences that caused filter to activate
            for pos in activated_positions:
                subseq = seq[pos:pos+filt_width]
                #subseq_tensor = torch.tensor(u.one_hot_encode(subseq)).view(-1,filt_width,4).permute(0,2,1).squeeze(0)
                subseq_tensor = torch.tensor(u.one_hot_encode(subseq)).permute(0,2,1).squeeze(0)
                # OHE FIX??
                
                # add this subseq to the pwm count for this filter
                filter_pwms[filt_id] += subseq_tensor
            
    return filter_pwms


def view_filters_and_logos(model_weights,filter_activations, num_cols=8):
    
    assert(model_weights[0].shape[0] == len(filter_activations))
    # make sure the model weights agree with the number of filters
    num_filts = len(filter_activations)
    num_rows = int(np.ceil(num_filts/num_cols))*2+1 # not sure why +1 is needed... complained otherwise
    
    plt.figure(figsize=(20, 17))

    j=0 # use to make sure a filter and it's logo end up vertically paired
    for i, filter in enumerate(model_weights[0]):
        if (i)%num_cols == 0:
            j += num_cols
    #     print('i:', i)
    #     print('j:', j)
    #     print('i%8 == 0', i%8 == 0)
    # #     print('i+1%9 =?', (i+1)%9)
    #     print("i+j+1=", i+j+1)
    #     print("i+j+1+4=", i+j+1+8)
    #     print("*******")

        # display raw filter
        ax1 = plt.subplot(num_rows, num_cols, i+j+1)
        ax1.imshow(filter.detach(), cmap='gray')
        ax1.set_yticks(np.arange(4))
        ax1.set_yticklabels(['A', 'C', 'G','T'])
        ax1.set_xticks(np.arange(3))
        ax1.set_title(f"Filter {i}")

        # display sequence logo
        ax2 = plt.subplot(num_rows, num_cols, i+j+1+num_cols)
        filt_df = pd.DataFrame(filter_activations[i].T.numpy(),columns=['A','C','G','T'])
        filt_df_info = logomaker.transform_matrix(filt_df,from_type='counts',to_type='information')
        logo = logomaker.Logo(filt_df_info,ax=ax2)
        ax2.set_ylim(0,2)
        ax2.set_title(f"Filter {i}")

    plt.tight_layout()