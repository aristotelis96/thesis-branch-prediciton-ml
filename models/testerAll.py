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
import numpy as np
import gzip

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
            self.fc = nn.Linear(out_dim, 2) # nn.Linear(out_dim*200, 2) 
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

        #data = self.E[seq.data.type(torch.long)]
        #seq = data.to(torch.device("cuda:0"))

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
model = None
def main():
    ## GPU/CPU ##
    device_idx = 0
    device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    ## Parameters ##
    n_class = 128
    num_layers = 1
    normalization = True
    input_dim = 1
    rnn_layer = 'lstm'

    # Name of pt model
    modelFolder = "./pt/lstm/encodedPC/dekadikosArithmos/"
    modelName = "lstm128_lr1e-4_encodedPC_210B.pt"

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

    correct = 0
    total=0
    now = time.time()

    with gzip.open("../Datasets/600.perlbench_s-210B.champsimtrace.xz._.dataset_all.txt.gz", 'rt') as fp:      
        model.eval()
        try:        
            line=""
            while True:
                sample = torch.zeros((1,200,1), dtype=torch.float64)
                if "--- H2P ---" not in line:
                    for line in fp:
                        if "--- H2P ---" in line:
                            break            
                label = np.float64(fp.readline().split(" ")[1])
                history=0            
                for line in fp:
                    if "--- H2P ---" in line or "\n"==line or line.startswith("Warmup complete"):                    
                        break
                    [ip, taken] = line.split(" ")
                    pc = int(ip)
                    encodedPC = (pc & 0b1111111) << 1
                    encodedPC += int(taken)
                    sample[0][history][0] = encodedPC
                    history+=1            
                if(model(sample.float()).argmax(axis=1)==label):
                    correct+=1
                total+=1
                if(total==1000000):
                    break
            print(correct,total, 100*correct/total)
        except Exception as e:
            print("ERROR" ,e)
    end = time.time()
    print("total time taken to train: ", end-now)        
main()
if __name__ == '__main__':
    main