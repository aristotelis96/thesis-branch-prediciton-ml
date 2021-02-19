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
import pprint

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
       # xFull = xFull.reshape(len(xFull), 32,8,200).to(device)
        
        h1 = self.c1(xFull)
        h1a = self.tahn(h1)
        h1a = h1a.reshape(len(h1a),self.historyLen*self.numFilters)
        out = self.l2(h1a)        
        out = self.sigmoid2(out)
        return out



class RNNLayer(nn.Module):
    def __init__(self,  input_dim, out_dim, num_layers, init_weights = True, batch_first=True, rnn_layer = 'gru', normalization = False):
        super().__init__()        

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.rnn_layer = rnn_layer
        
        self.E = torch.eye(self.input_dim, dtype=torch.float32)        

        self.embedding1 = nn.Embedding(128,64)
        self.embedding2 = nn.Embedding(2,64)

        if rnn_layer == 'lstm':
            self.rnn = nn.LSTM(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
            self.fc = nn.Linear(out_dim*200, 2)
        elif rnn_layer == 'gru':
            self.rnn = nn.GRU(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
        elif rnn_layer =='transformer':
            #self.inpLinear = nn.Linear(2, self.input_dim)
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
        lay1=self.embedding1(seq.data[:,:,0].type(torch.long).to(device))        
        lay2=self.embedding2(seq.data[:,:,1].type(torch.long).to(device))        
        seq = torch.cat((lay1,lay2),2)

        if self.rnn_layer == 'transformer':
            #seq = self.inpLinear(seq)
            self.rnn_out = self.rnn(seq)                   
            #self.rnn_out = self.rnn_out[:,-1,:]  
            self.rnn_out = self.rnn_out.reshape(len(self.rnn_out), 200*self.input_dim)            
        elif self.rnn_layer =='lstm':
            self.rnn_out, (self.h, self.c) = self.rnn(seq)                      
            self.rnn_out = self.rnn_out.flatten(1)                     
        elif self.rnn_layer == 'gru': 
            self.rnn_out, self.h = self.rnn(seq)

        if self.normalization:
            self.rnn_out = self.normalization(self.rnn_out)
        
        out = self.fc(self.rnn_out)   

        ## out = self.fc(self.rnn_out.view(bsize,-1))        
        return self.softmax(out)


device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
model = None
allBranch = {}

def main(outputName, mode):
    ## CNN or LSTM
    mode = mode ## SOOOS REMEMBER TO CHANGE READING SEE LINE: 236    
    bench = "557.xz"
    
    ## GPU/CPU ##
    device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")


    # Name of pt model
    modelFolder = "./specificBranch/"+bench+"/models/"+mode+"/"
    models = os.listdir(modelFolder)
    modelsDict = {}
    
    if (mode=="CNN"):
        ## Parameters ##
        tableSize = 256
        numFilters = 2
        historyLen = 200
        for modelN in models:
            modelName = modelN
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
                model.eval()
            except:
                print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
                exit()
            modelsDict[int(modelN[3:-3])] = model

    if (mode=="LSTM"):
        ## Parameters ##
        n_class = 128
        num_layers = 1
        normalization = False
        input_dim = 128
        rnn_layer = 'lstm'

        for modelN in models:
            modelName = modelN
            modelName = modelFolder + modelName
            ## Model 
            model = RNNLayer(rnn_layer=rnn_layer, input_dim=input_dim, out_dim=n_class, num_layers=num_layers, normalization=normalization).to(device)

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
                model.eval()
            except:
                print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
                exit()
            modelsDict[int(modelN[4:-3])] = model

    if ('encodePCList' in checkpoint['input_Data']):
        encodePCList = checkpoint['input_Data']['encodePCList']    

    correct = 0
    total=0
    now = time.time()

    bench = "../Datasets/myTraces/"+bench+"/557.xz_ref-cld-233B.champsimtrace.xz._.dataset_all.txt.gz"
    print(bench)
    with gzip.open(bench, 'rt') as fp:      
        #model.eval()
        try:        
            line=""
            if "--- H2P ---" not in line:
                for line in fp:
                    if "--- H2P ---" in line:
                        break            
            while True:
                if (mode=="CNN"): sample = torch.zeros((1,200), dtype=torch.float64) 
                if(mode=="LSTM"): sample = torch.zeros((1,200,2), dtype=torch.float64) #LSTM
                line = fp.readline()            
                if "--- H2P ---" in line or "\n"==line or line.startswith("Warmup"):                    
                    continue
                [ipH2P, label] = np.float64(line.split(" "))                
                history=0            
                for line in fp:
                    if "--- H2P ---" in line or "\n"==line or line.startswith("Warmup"):                    
                        break
                    [ip, taken] = line.split(" ")
                    pc = int(ip)
                    if(mode=="CNN"):
                        encodedPC = (pc & 0b1111111) << 1
                        encodedPC += int(taken)
                        sample[0][history] = encodedPC
                    if(mode=="LSTM"):
                        encodedPC = (pc & 0b1111111)
                        sample[0][history][0] = encodedPC
                        sample[0][history][1] = int(taken)
                    history+=1            
                ipH2P = int(ipH2P)                
                if ipH2P not in allBranch:
                    allBranch[ipH2P] = {'total': 1, 'correct' : 0, 'acc': 0}
                else:
                    allBranch[ipH2P]['total']+=1
                if(ipH2P in modelsDict):
                    prediction = False
                    if(mode=="CNN"):
                        prediction = torch.round(modelsDict[ipH2P](sample.float()))==label
                    elif (mode=="LSTM"):
                        prediction = (modelsDict[ipH2P](sample.float()).argmax(axis=1).cpu() == label)
                    if(prediction):
                        correct+=1
                        allBranch[ipH2P]['correct']+=1
                        allBranch[ipH2P]['acc'] = allBranch[ipH2P]['correct']/allBranch[ipH2P]['total']                
                total+=1
                if(total%500000==0):
                    print(correct,total, 100*correct/total)
                    p = pprint.PrettyPrinter()
                    p.pprint(allBranch)
                    torch.save(allBranch, outputName)
            print(correct,total, 100*correct/total)
        except Exception as e:
            print("ERROR" ,e)
            print(correct,total, 100*correct/total)
            p.pprint(allBranch)
            torch.save(allBranch, outputName)

    end = time.time()
    print("total time taken to check: ", end-now)        
if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print("usage: script 'outputName' 'MODE:LSTM/CNN'")
        exit()
    main(sys.argv[1], sys.argv[2])