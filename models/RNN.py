import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.utils.data import DataLoader
import time
import read
from read import BranchDataset
import os

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
    def __init__(self,  input_dim, out_dim, num_layers, init_weights = True, batch_first=True, rnn_layer = 'gru', normalization = False):
        super().__init__()
      
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.rnn_layer = rnn_layer
        
        
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

        if (normalization):
            self.normalization = nn.BatchNorm1d(out_dim)

        self.fc = nn.Linear(out_dim, 2) # nn.Linear(out_dim*200, 2)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, seq):
        bsize = seq.size(0)        

        if self.rnn_layer == 'transformer':
            self.rnn_out = self.rnn(seq)
        elif self.rnn_layer =='lstm':
            self.rnn_out, (self.h, self.c) = self.rnn(seq)
            self.rnn_out = self.h[-1]            
        elif self.rnn_layer == 'gru': 
            self.rnn_out, self.h = self.rnn(seq)
        
        if normalization:
            self.rnn_out = self.normalization(self.rnn_out)
            
        out = self.fc(self.rnn_out)   
        ## out = self.fc(self.rnn_out.view(bsize,-1))        
        return self.softmax(out)

## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

n_class = 64
num_layers = 1
normalization = False
input_dim = 2
rnn_layer = 'lstm'

ContinueFromCheckpoint = False

epochStart = 0
epochEnd = 100

input_bench = ["600.perlbench_s-1273B.champsimtrace.xz._.dataset_unique.txt.gz"]
## Model 
model = RNNLayer(rnn_layer=rnn_layer, input_dim=input_dim, out_dim=n_class, num_layers=num_layers, normalization=normalization).to(device)

print(model)
## TRAIN/TEST DATALOADER
# Parameters
paramsTrain = {'batch_size': 64,
          'shuffle': True,

          'num_workers': 3}
paramsValid = {'batch_size': 10000,
          'shuffle': False,
          'num_workers': 4}

print("Loading TrainDataset")
print("Loading ValidationDataset")
train, valid = read.readFileList(input_bench, 100,60000)

training_set, validation_set = BranchDataset(train), BranchDataset(valid)

train_loader = DataLoader(training_set, **paramsTrain)
valid_loader = DataLoader(validation_set, **paramsValid)

##
## Optimize 
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate/10)

total_Train_loss = []
total_Train_accuracy = []
total_Validation_loss = []
total_Validation_accuracy = []


if ContinueFromCheckpoint:
    checkpoint = torch.load("checkpoint.pt")
    epochStart = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    total_Train_loss = checkpoint['total_Train_loss']
    total_Train_accuracy = checkpoint['total_Train_accuracy']
    total_Validation_loss = checkpoint['total_Validation_loss']
    total_Validation_accuracy = checkpoint['total_Validation_accuracy']
else:
    try:
        os.remove("checkpoint.pt")
    except:
        pass


#TRAINING
now = time.time()
for epoch in range(epochStart, epochEnd):
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
    train_acc = correct/float(len(training_set))
    train_loss = running_loss/float(len(train_loader))
    total_Train_accuracy.append(train_acc)
    total_Train_loss.append(train_loss)
    print("Epoch: ", epoch, "train loss:", train_loss, " acc:", train_acc)
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
    total_Validation_loss.append(epochLoss)

    epochAccuracy = float(correct)/float(len(validation_set))
    total_Validation_accuracy.append(epochAccuracy)
    print("Epoch: ", epoch, "validation: loss:", epochLoss ,"acc:" , epochAccuracy)
    if(epochAccuracy>0.99):
        break
        # ADD METRIC (ACCURACY? Note batching)
    #print(loss_values)
end = time.time()
print("total time taken to train: ", end-now)
## SAVE A CHECKPOINT as "checkpoint.pt"
torch.save({
        'epoch': epoch,
        'model_architecture': str(model),
        'model_specifics': {
            'n_class' : n_class,
            'num_layers' : num_layers,
            'normalization' : normalization,
            'input_dim' : input_dim,
            'rnn_layer' : rnn_layer
        },
        'input_Data': {
            'input_bench': input_bench,
            'type': '1-hot sequence-200 only Taken/NotTaken, no program counter',
            'batch_size': paramsTrain['batch_size']
        },
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_Validation_accuracy': total_Validation_accuracy,
        'total_Validation_loss': total_Validation_loss,
        'total_Train_loss': total_Train_loss,
        'total_Train_accuracy': total_Train_accuracy
    }, "checkpoint.pt")

print("Finish")
