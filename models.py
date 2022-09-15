# models.py
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np


class DNA_Linear_Deep(nn.Module):
    def __init__(
        self, 
        seq_len,
        h0_size=24,
        h1_size=24,
        num_classes=3
    ):
        super().__init__()
        self.seq_len = seq_len
        
        self.lin = nn.Sequential(
            nn.Linear(4*seq_len, h1_size),
            nn.ReLU(inplace=True),
            nn.Linear(h0_size, h1_size),
            nn.ReLU(inplace=True),
            nn.Linear(h1_size, num_classes), # 3 for 3 classes
            #nn.Softmax(dim=1)
        )

    def forward(self, xb):
        # Linear wraps up the weights/bias dot product operations
        # reshape to flatten sequence dimension
        xb = xb.view(xb.shape[0],self.seq_len*4)
        out = self.lin(xb)
        #print("Lin out shape:", out.shape)
        return out
    
class DNA_CNN(nn.Module):
    def __init__(self,
                 seq_len,
                 num_filters=31,
                 kernel_size=3,
                 num_classes=3
                ):
        super().__init__()
        self.seq_len = seq_len
        self.lin_nodes = num_filters*(seq_len-kernel_size+1)
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=(4,kernel_size)),
            # ^^ changed from 4 to 1 channel??
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.lin_nodes, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes),
            #nn.Softmax(dim=1)
            
        ) 

    def forward(self, xb):
        # reshape view to batch_ssize x 4channel x seq_len
        # permute to put channel in correct order
        #xb = xb.view(-1,self.seq_len,4).permute(0,2,1) 
        #xb = xb.permute(0,2,1) 
        # OHE FIX??
        xb = xb.permute(0,2,1).unsqueeze(1)
        # ^^ Conv2D input fix??
        #print(xb.shape)
        out = self.conv_net(xb)
        #print("CNN out shape:",out.shape)
        return out

class Kmer_Linear(nn.Module):
    def __init__(self,
    			 num_kmers,
    			 h1_size=100,
    			 h2_size=10,
    			 num_classes=3
    			 ):
        super().__init__()
        
        # some arbitrary arch of a few linear layers
        self.lin = nn.Sequential(
            nn.Linear(num_kmers, h1_size), ## TODO: Can this be combined? Bring num_kmers outside?
            nn.ReLU(inplace=True),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(inplace=True),
            nn.Linear(h2_size, num_classes),
        )
        
        
    def forward(self, xb):
        out = self.lin(xb)
        #print("Lin out shape:", out.shape)
        return out

class TINKER_DNA_CNN(nn.Module):
    def __init__(self,
                 seq_len,
                 num_filters0=32,
                 num_filters1=32,
                 kernel_size0=8,
                 kernel_size1=8,
                 conv_pool_size0=1,
                 conv_pool_size1=1,
                 fc_node_num0 = 10,
                 fc_node_num1 = 10
                ):
        super().__init__()
        
        self.seq_len = seq_len
        
        # calculation for number of linear nodes need to come after final conv layer
        linear_node_num = int(np.floor((seq_len - kernel_size0 + 1)/conv_pool_size0))
        linear_node_num = int(np.floor((linear_node_num - kernel_size1 + 1)/conv_pool_size1))
        linear_node_num = linear_node_num*num_filters1
        #linear_node_num = linear_node_num*num_filters0
        
        self.conv_net = nn.Sequential(
            # Conv layer 0
            nn.Conv2d(1, num_filters0, kernel_size=(4,kernel_size0)),
            # ^^ changed from 4 to 1 channel??
            nn.ReLU(),
            nn.MaxPool2d((1,conv_pool_size0)), # def stride = kernel_size
            nn.Dropout(0.5),

            # Conv layer 1
            nn.Conv2d(num_filters0, num_filters1, kernel_size=(1,kernel_size1)),
            nn.ReLU(),
            
            nn.Flatten(),
            # Fully connected layer 0
            nn.Linear(linear_node_num, fc_node_num0),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Fully connected layer 0
#             nn.Linear(fc_node_num0, fc_node_num1),
#             nn.ReLU(),
            # final prediction
            nn.Linear(fc_node_num0, 3),
            #nn.Softmax(dim=1)
        ) 

    def forward(self, xb):
        # reshape view to batch_ssize x 4channel x seq_len
        # permute to put channel in correct order
        
        #xb = xb.permute(0,2,1) 
        # OHE FIX??
        
        xb = xb.permute(0,2,1).unsqueeze(1)
        # ^^ Conv2D input fix??
        
        out = self.conv_net(xb)
        return out

class DNA_2CNN(nn.Module):
    def __init__(self,
                 seq_len,
                 num_filters1=32,
                 num_filters2=32,
                 kernel_size1=8,
                 kernel_size2=8,
                 conv_pool_size1=1, # default no pooling
                 conv_pool_size2=1,
                 fc_node_num1 = 10,
                 fc_node_num2 = 10,
                 dropout1 = 0.2,
                 dropout2 = 0.2,
                ):
        super().__init__()
        
        self.seq_len = seq_len
        
        # calculation for number of linear nodes need to come after final conv layer
        linear_node_num = int(np.floor((seq_len - kernel_size1 + 1)/conv_pool_size1))
        linear_node_num = int(np.floor((linear_node_num - kernel_size2 + 1)/conv_pool_size2))
        linear_node_num = linear_node_num*num_filters2
        #linear_node_num = linear_node_num*num_filters1
        print("final linear_node_num:", linear_node_num)
        
        self.conv_net = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(1, num_filters1, kernel_size=(4,kernel_size1)),
            # ^^ changed from 4 to 1 channel??
            nn.ReLU(),
            nn.MaxPool2d((1,conv_pool_size1)), # def stride = kernel_size
            nn.Dropout(dropout1),

            # Conv layer 2
            nn.Conv2d(num_filters1, num_filters2, kernel_size=(1,kernel_size2)),
            nn.ReLU(),
            nn.Dropout(dropout2),
            
            nn.Flatten(),
            # Fully connected layer 1
            nn.Linear(linear_node_num, fc_node_num1),
            nn.ReLU(),
            # Fully connected layer 2
#             nn.Linear(fc_node_num1, fc_node_num2),
#             nn.ReLU(),
            # final prediction
            nn.Linear(fc_node_num1, 1),
        ) 

    def forward(self, xb):
        # reshape view to batch_ssize x 4channel x seq_len
        # permute to put channel in correct order
        
        #xb = xb.permute(0,2,1) 
        # OHE FIX??
        
        xb = xb.permute(0,2,1).unsqueeze(1)
        # ^^ Conv2D input fix??
        
        out = self.conv_net(xb)
        return out

# can this be combined with single task model?
class DNA_2CNN_Multi(nn.Module):
    def __init__(self,
                 seq_len,
                 n_tasks,
                 num_filters1=32,
                 num_filters2=32,
                 kernel_size1=8,
                 kernel_size2=8,
                 conv_pool_size1=1, # default no pooling
                 conv_pool_size2=1,
                 fc_node_num1 = 10,
                 fc_node_num2 = 10,
                 dropout1 = 0.2,
                 dropout2 = 0.2,
                ):
        super().__init__()
        
        self.seq_len = seq_len
        self.n_tasks = n_tasks
        
        # calculation for number of linear nodes need to come after final conv layer
        linear_node_num = int(np.floor((seq_len - kernel_size1 + 1)/conv_pool_size1))
        linear_node_num = int(np.floor((linear_node_num - kernel_size2 + 1)/conv_pool_size2))
        linear_node_num = linear_node_num*num_filters2
        #linear_node_num = linear_node_num*num_filters1

        self.conv_net = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(1, num_filters1, kernel_size=(4,kernel_size1)),
            # ^^ changed from 4 to 1 channel??
            nn.ReLU(),
            nn.MaxPool2d((1,conv_pool_size1)), # def stride = kernel_size
            nn.Dropout(dropout1),

            # Conv layer 2
            nn.Conv2d(num_filters1, num_filters2, kernel_size=(1,kernel_size2)),
            nn.ReLU(),
            nn.Dropout(dropout2),
            
            nn.Flatten(),
            # Fully connected layer 1
            nn.Linear(linear_node_num, fc_node_num1),
            nn.ReLU(),
            # Fully connected layer 2
#             nn.Linear(fc_node_num1, fc_node_num2),
#             nn.ReLU(),
            # final prediction
            nn.Linear(fc_node_num1, n_tasks),
        ) 

    def forward(self, xb):
        # reshape view to batch_ssize x 4channel x seq_len
        # permute to put channel in correct order
        
        #xb = xb.permute(0,2,1) 
        # OHE FIX??
        
        xb = xb.permute(0,2,1).unsqueeze(1)
        # ^^ Conv2D input fix??
        
        out = self.conv_net(xb)
        return out

# can this be combined with single task model?
class DNA_2CNN_2FC_Multi(nn.Module):
    def __init__(self,
                 seq_len,
                 n_tasks,
                 num_filters1=32,
                 num_filters2=32,
                 kernel_size1=8,
                 kernel_size2=8,
                 conv_pool_size1=1, # default no pooling
                 conv_pool_size2=1,
                 fc_node_num1 = 10,
                 fc_node_num2 = 10,
                 dropout1 = 0.2,
                 dropout2 = 0.2,
                ):
        super().__init__()
        
        self.seq_len = seq_len
        self.n_tasks = n_tasks
        
        # calculation for number of linear nodes need to come after final conv layer
        linear_node_num = int(np.floor((seq_len - kernel_size1 + 1)/conv_pool_size1))
        linear_node_num = int(np.floor((linear_node_num - kernel_size2 + 1)/conv_pool_size2))
        linear_node_num = linear_node_num*num_filters2
        #linear_node_num = linear_node_num*num_filters1

        self.conv_net = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(1, num_filters1, kernel_size=(4,kernel_size1)),
            # ^^ changed from 4 to 1 channel??
            nn.ReLU(),
            nn.MaxPool2d((1,conv_pool_size1)), # def stride = kernel_size
            nn.Dropout(dropout1),

            # Conv layer 2
            nn.Conv2d(num_filters1, num_filters2, kernel_size=(1,kernel_size2)),
            nn.ReLU(),
            nn.Dropout(dropout2),
            
            nn.Flatten(),
            # Fully connected layer 1
            nn.Linear(linear_node_num, fc_node_num1),
            nn.ReLU(),
            # Fully connected layer 2
            nn.Linear(fc_node_num1, fc_node_num2),
            nn.ReLU(),
            # final prediction
            nn.Linear(fc_node_num2, n_tasks),
        ) 

    def forward(self, xb):
        # reshape view to batch_ssize x 4channel x seq_len
        # permute to put channel in correct order
        
        #xb = xb.permute(0,2,1) 
        # OHE FIX??
        
        xb = xb.permute(0,2,1).unsqueeze(1)
        # ^^ Conv2D input fix??
        
        out = self.conv_net(xb)
        return out

class DNA_LSTM(nn.Module):
    def __init__(self,
                 seq_len,
                 device,
                 hidden_dim=100,
                 num_classes=3,
                ):
        super().__init__()
        self.seq_len = seq_len
        self.device = device

        self.hidden_dim = hidden_dim
        self.hidden = None # when initialized, should be tuple of (hidden state, cell state)
        
        self.rnn = nn.LSTM(4, hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
            

    
    def init_hidden(self,batch_size):
        # initialize hidden and cell states with 0s
        self.hidden =  (torch.zeros(1, batch_size, self.hidden_dim).to(self.device), 
                        torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
        return self.hidden
        #hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
    

    def forward(self, xb,verbose=False):
        if verbose:
            print("original xb.shape:", xb.shape)
            print(xb) # 11 x 32
        
        # make the one-hot nucleotide vectors group together
        xb = xb.view(-1,self.seq_len,4) 
        if verbose:
            print("re-viewed xb.shape:", xb.shape) # >> 11 x 8 x 4
            print(xb)

        # ** Init hidden/cell states?? **
        batch_size = xb.shape[0]
        if verbose:
            print("batch_size:",batch_size)
        (h,c) = self.init_hidden(batch_size)
         
        # *******
        
        lstm_out, self.hidden = self.rnn(xb, (h,c)) # should this get H and C?
        if verbose:
            #print("lstm_out",lstm_out)
            print("lstm_out shape:",lstm_out.shape) # >> 11, 8, 10
            print("lstm_out[-1] shape:",lstm_out[-1].shape) # >> 8 x 10
            print("lstm_out[-1][-1] shape:",lstm_out[-1][-1].shape) # 10

            print("hidden len:",len(self.hidden)) # 2
            print("hidden[0] shape:", self.hidden[0].shape) # >> 1 x 11 x 10
            print("hidden[0][-1] shape:", self.hidden[0][-1].shape) # >> 11 X 10
            print("hidden[0][-1][-1] shape:", self.hidden[0][-1][-1].shape) # >> 10

            print("*****")
            # These vectors should be the same, right?
            A = lstm_out[-1][-1]
            B = self.hidden[0][-1][-1]
            print("lstm_out[-1][-1]:",A)
            print("self.hidden[0][-1][-1]",B)
            print("==?", A==B)
            print("*****")
        
        # attempt to get the last layer from each last position of 
        # all seqs in the batch? IS this the right thing to get?
        last_layer = lstm_out[:,-1,:] # This is 11X10... and it makes FC out 11X1, which is what I want?
        #last_layer = lstm_out[-1][-1].unsqueeze(0) # this was [10X1]? led to FC outoput being [1]?
        if verbose:
            print("last layer:", last_layer.shape)

        out = self.fc(last_layer) 
        if verbose:
            print("LSTM->FC out shape:",out.shape)   
                                                
        return out


class DNA_CNNLSTM(nn.Module):
    def __init__(self,
                 seq_len,
                 device,
                 hidden_dim=100,
                 num_filters=32,
                 kernel_size=6,
                 num_classes=3):
        super().__init__()
        self.seq_len = seq_len
        self.device = device
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, num_filters, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
        ) 
        
        self.hidden_dim = hidden_dim
        self.hidden = None # when initialized, should be tuple of (hidden state, cell state)
        
        self.rnn = nn.LSTM(num_filters, hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
            
    def init_hidden(self,batch_size):
        # initialize hidden and cell states with 0s
        self.hidden =  (torch.zeros(1, batch_size, self.hidden_dim).to(self.device), 
                        torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
        return self.hidden
        #hidden_state = torch.randn(n_layers, batch_size, hidden_dim)

    def forward(self, xb, verbose=False):
        # reshape view to batch_ssize x 4channel x seq_len
        # permute to put channel in correct order
        xb = xb.view(-1,self.seq_len,4).permute(0,2,1) 
        if verbose:
            print("xb reviewed shape:",xb.shape)

        cnn_out = self.conv_net(xb)
        if verbose:
            print("CNN out shape:",cnn_out.shape)
        cnn_out_perm = cnn_out.permute(0,2,1)
        if verbose:
            print("CNN permute out shape:",cnn_out_perm.shape)
        
        batch_size = xb.shape[0]
        if verbose:
            print("batch_size:",batch_size)
        (h,c) = self.init_hidden(batch_size)
        
        lstm_out, self.hidden = self.rnn(cnn_out_perm, (h,c)) # should this get H and C?
        
        last_layer = lstm_out[:,-1,:] # This is 11X10... and it makes FC out 11X1, which is what I want?
        if verbose:
            print("last layer:", last_layer.shape)

        out = self.fc(last_layer) 
        if verbose:
            print("LSTM->FC out shape:",out.shape)        
        
        return out