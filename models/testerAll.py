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


class TwoLayerFullPrecisionBPCNN(nn.Module):
    def __init__(self, tableSize=256, numFilters=2, historyLen=200):
        super().__init__()
        self.historyLen= historyLen
        self.numFilters = numFilters
        #embed via identity matrix:
        self.E = torch.eye(tableSize, dtype=torch.float32)
        
        #convolution layer, 1D in effect; Linear Layer
        self.c1 = nn.Conv2d(tableSize, numFilters, (1,1))        
        self.tahn = nn.Tanh()
        self.l2 = nn.Linear(historyLen*numFilters, 1)        
        self.sigmoid2 = nn.Sigmoid()        

    def forward(self, seq):
        #Embed by expanding sequence of ((IP << 7) + dir ) & (255)
        # integers into 1-hot history matrix during training
        
        xFull = self.E[seq.data.type(torch.long)]
        xFull = torch.unsqueeze(xFull, 1)        
        xFull = xFull.permute(0,3,1,2).to(device)
        
        h1 = self.c1(xFull)
        h1a = self.tahn(h1)
        h1a = h1a.reshape(len(h1a),self.historyLen*self.numFilters)
        out = self.l2(h1a)        
        out = self.sigmoid2(out)
        return out
device = torch.device("cpu")
model = None
allBranch = {}

def main():
    ## GPU/CPU ##
    device = torch.device("cpu")
    ## Parameters ##
    tableSize = 256
    numFilters = 2
    historyLen = 200

    # Name of pt model
    modelFolder = "./"
    modelName = "checkpointCNN.pt"

    modelName = modelFolder + modelName
    ## Model 
    model = TwoLayerFullPrecisionBPCNN(tableSize=tableSize, numFilters=numFilters, historyLen=historyLen).to(device)

    print(model)
    ## TEST DATALOADER
    # Parameters
    paramsValid = {'batch_size': 64,
            'shuffle': False,
            'num_workers': 4}

   

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
            if "--- H2P ---" not in line:
                for line in fp:
                    if "--- H2P ---" in line:
                        break            
            while True:
                sample = torch.zeros((1,200), dtype=torch.float64)                
                [ipH2P, label] = np.float64(fp.readline().split(" "))
                history=0            
                for line in fp:
                    if "--- H2P ---" in line or "\n"==line or line.startswith("Warmup complete"):                    
                        break
                    [ip, taken] = line.split(" ")
                    pc = int(ip)
                    encodedPC = (pc & 0b1111111) << 1
                    encodedPC += int(taken)
                    sample[0][history] = encodedPC
                    history+=1            
                ipH2P = int(ipH2P)
                if ipH2P not in allBranch:
                    allBranch[ipH2P] = {'total': 1, 'correct' : 0, 'acc': 0}
                else:
                    allBranch[ipH2P]['total']+=1
                if(torch.round(model(sample.float()))==label):
                    correct+=1
                    allBranch[ipH2P]['correct']+=1
                    allBranch[ipH2P]['acc'] = allBranch[ipH2P]['correct']/allBranch[ipH2P]['total']
                total+=1
                if(total%5000==0):
                    print(correct,total, 100*correct/total)
                    print(allBranch)
                    break
            print(correct,total, 100*correct/total)
        except Exception as e:
            print("ERROR" ,e)
            print(correct,total, 100*correct/total)
    end = time.time()
    print("total time taken to check: ", end-now)        
if __name__ == '__main__':
    main()