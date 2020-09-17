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
        
        self.E = torch.eye(256, dtype=torch.float32)        

        if rnn_layer == 'lstm':
            self.rnn = nn.LSTM(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
        elif rnn_layer == 'gru':
            self.rnn = nn.GRU(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
        elif rnn_layer =='transformer':
            self.rnn = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.out_dim)
            self.fc = nn.Linear(input_dim*200, 2) # nn.Linear(out_dim*200, 2)
        else:
            raise NotImplementedError(rnn_layer)
        
        if init_weights:
            self.rnn = init_lstm(self.rnn, self.out_dim)

        if (normalization):
            self.normalization = nn.BatchNorm1d(out_dim)
        else:
            self.normalization = None

        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, seq):
        bsize = seq.size(0)        

        data = self.E[seq.data.type(torch.long)]
        seq = data.to(torch.device("cuda:0"))

        if self.rnn_layer == 'transformer':
            #seq = self.inpLinear(seq)
            self.rnn_out = self.rnn(seq)                   
            #self.rnn_out = self.rnn_out[:,-1,:]  
            self.rnn_out = self.rnn_out.reshape(len(self.rnn_out), 200*256)
        elif self.rnn_layer =='lstm':
            self.rnn_out, (self.h, self.c) = self.rnn(seq) 
            self.rnn_out = self.h[-1]            
        elif self.rnn_layer == 'gru': 
            self.rnn_out, self.h = self.rnn(seq)

        if self.normalization:
            self.rnn_out = self.normalization(self.rnn_out)
        
        out = self.fc(self.rnn_out)   
        ## out = self.fc(self.rnn_out.view(bsize,-1))        
        return self.softmax(out)

def main():
    ## GPU/CPU ##
    device_idx = 0
    device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")

    ## Parameters ##
    n_class = 256
    num_layers = 2
    normalization = False
    input_dim = 256
    rnn_layer = 'transformer'

    # Name of pt model
    modelFolder = "./pt/transformer/"
    modelName = "transformer_256.pt"

    modelName = modelFolder + modelName
    ## Model 
    model = RNNLayer(rnn_layer=rnn_layer, input_dim=input_dim, out_dim=n_class, num_layers=num_layers, normalization=normalization).to(device)
    criterion = nn.CrossEntropyLoss()

    print(model)
    ## TEST DATALOADER
    # Parameters
    paramsValid = {'batch_size': 64,
            'shuffle': False,
            'num_workers': 4}

    # Benchmark
    input_bench = ["600.perlbench_s-1273B.champsimtrace.xz._.dataset_unique.txt.gz"]
    startSample, endSample = 100, 600000
    encodePCList = True
    loadPt = True

    try:
        print(modelName)
        checkpoint = torch.load(modelName)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
        exit()

    if ('encodePCList' in checkpoint['input_Data']):
        encodePCList = checkpoint['input_Data']['encodePCList']    

    print("Loading ValidationDataset")
    
    if(loadPt):
        valid = torch.load("../Datasets/perlbench_1273B_valid600K.pt")
    else:
        _, valid = read.readFileList(input_bench, startSample,endSample, ratio=0.0)
        torch.save(valid, '600.perlbench_1273B_valid600K.pt')
    
    
    validation_set = BranchDataset(valid, encodePCList=encodePCList)

    print("len of validation set:", len(validation_set))

    valid_loader = DataLoader(validation_set, **paramsValid)
    print("-------")
    #print("Epoch : " + str(epoch))
    loss_values = []
    running_loss = 0.0
    correct = 0.0
    values = []
    for X_val, Validlabels in valid_loader:
        model.eval() 

        X_val = X_val.to(device)
        Validlabels = Validlabels.to(device)

        outputs = model(X_val.float())

        loss = criterion(outputs, Validlabels.long())

        loss_values.append(loss.item())    
        values.push(outputs)
        correct += (outputs.argmax(axis=1).cpu() == Validlabels.cpu()).sum()
    epochLoss = float(sum(loss_values))/float(len(valid_loader))
    epochAccuracy = float(correct)/float(len(validation_set))
    print("validation: loss:", epochLoss ,"acc:" , epochAccuracy)
    print("Finish")


if __name__ == '__main__':
    main()