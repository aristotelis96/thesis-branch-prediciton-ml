import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.utils.data import DataLoader, random_split

import read
from read import BranchDataset

def init_lstm(lstm, lstm_hidden_size, forget_bias=2):
    for name,weights in lstm.named_parameters():
        if "bias_hh" in name:
            #weights are initialized 
            #(b_hi|b_hf|b_hg|b_ho), 
            weights[lstm_hidden_size:lstm_hidden_size*2].data.fill_(forget_bias)
        elif 'bias_ih' in name:
            #(b_ii|b_if|b_ig|b_io)
            pass
        elif "weight_hh" in name:
            torch.nn.init.orthogonal_(weights)
        elif 'weight_ih' in name:
            torch.nn.init.xavier_normal_(weights)
    return lstm


class RNNLayer(nn.Module):
    def __init__(self,  input_dim, out_dim, num_layers, init_weights = True, batch_first=True, rnn_layer = 'lstm'):
        super().__init__()
      
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.rnn_layer = rnn_layer
        
        self.softmax = nn.Softmax(dim=-1)
        if rnn_layer == 'lstm':
            self.rnn = nn.LSTM(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
        elif rnn_layer == 'gru':
            self.rnn = nn.GRU(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
        elif rnn_layer =='transformer':
            self.rnn = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=7)
        else:
            raise NotImplementedError(rnn_layer)
        
        if init_weights:
            self.rnn = init_lstm(self.rnn, self.out_dim)
        self.fc = nn.Linear(out_dim, 2) # nn.Linear(out_dim*100, 2)
        
        
    def forward(self, seq):
        bsize = seq.size(0)
        if self.rnn_layer == 'transformer':
            self.rnn_out = self.rnn(seq)
        else:
            self.rnn_out, (self.h, self.c) = self.rnn(seq)
            out = self.fc(self.h[-1])   
            ## out = self.fc(self.rnn_out.view(bsize,-1))
        return self.softmax(out)

## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

n_class = 64
## Random Input / Add Tensor of size [Sample, Hist Len, 2]
#x = torch.rand(1000,100,2).to(device)
## Model 
model = RNNLayer(input_dim=2, out_dim=n_class, num_layers=1).to(device)
print(model)
## TRAIN/TEST DATALOADER
# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 4}

print("Loading TrainDataset")
#train = torch.load("train100K.pt")
train = read.read("600.perlbench_s-1273B.champsimtrace.xz._.dataset_unique.txt.gz", 100, 50000)
print("Loading ValidationDataset")
#valid = torch.load("valid100K.pt")
valid = read.read("600.perlbench_s-1273B.champsimtrace.xz._.dataset_unique.txt.gz", 100000, 110000)

#train = (0 - 100) * torch.randn(10000, 100,2) + 100
#valid = (0 - 100) * torch.randn(1000, 100,2) + 100

training_set, validation_set = BranchDataset(train), BranchDataset(valid)

train_loader = DataLoader(training_set, **params)
valid_loader = DataLoader(validation_set, **params)

##
## Optimize 
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate/10)

total_loss = []
total_accuracy = []
for epoch in range(100):
    print("-------")
    #print("Epoch : " + str(epoch))
    loss_values = []
    running_loss = 0.0
    correct = 0.0
    for i, (X, labels) in enumerate(train_loader):
        model.train() 
        
        X = X.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(X.float())

        loss = criterion(outputs, labels.long()) 

        

        loss.backward()

        optimizer.step()
        # print statistics
        running_loss += loss.item()
        correct += float((outputs.argmax(axis=1).cpu() == labels.cpu()).sum())
        #if i % 500 == 499:    # print every 2000 mini-batches
         #   print('[%d, %5d] loss: %.3f acc: %.3f' %
          #        (epoch, i + 1, running_loss / 500, correct / 500.0))
           # running_loss = 0.0
            #correct = 0
    print("Epoch: ", epoch, "train loss:", running_loss/float(len(train_loader)), " acc:", correct/float(len(training_set)))
    #if(correct/float(len(training_set))>0.99):
        #break
    correct = 0
    for X_val, Validlabels in valid_loader:
        model.eval() 

        X_val = X_val.to(device)
        Validlabels = Validlabels.to(device)

        outputs = model(X_val.float())

        loss = criterion(outputs, Validlabels.long())

        loss_values.append(loss.item())    
    
        correct += (outputs.argmax(axis=1).cpu() == Validlabels.cpu()).sum()
    epochLoss = float(sum(loss_values))/float(len(valid_loader))
    total_loss.append(epochLoss)

    epochAccuracy = float(correct)/float(len(validation_set))
    total_accuracy.append(epochAccuracy)
    print("Epoch: ", epoch, "validation: loss:", epochLoss ,"acc:" , epochAccuracy)
    if(epochAccuracy>0.99):
        break
        # ADD METRIC (ACCURACY? Note batching)
    #print(loss_values)

        
## SAVE MODEL (if necessary ??)
