import platform
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
import branchnet.model as BranchNetModel
import yaml
from pathlib import Path
import gc
from collections import deque

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

        self.embedding1 = nn.Embedding(256,self.input_dim)
        #self.embedding2 = nn.Embedding(2,64)

        if rnn_layer == 'lstm':
            self.rnn = nn.LSTM(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
            self.fc = nn.Linear(out_dim*200, 2)
        elif rnn_layer == 'gru':
            self.rnn = nn.GRU(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
        elif rnn_layer =='transformer':
            #self.inpLinear = nn.Linear(2, self.input_dim)
            #self.rnn = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.out_dim, dim_feedforward=16)
            self.rnn = nn.Transformer(d_model=self.input_dim, nhead=self.out_dim, num_encoder_layers=self.num_layers, num_decoder_layers=self.num_layers, dim_feedforward=512,dropout=0.5)
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
        #lay1=self.embedding1(seq.data[:,:,0].type(torch.long).to(device))        
        #lay2=self.embedding2(seq.data[:,:,1].type(torch.long).to(device))        
        #seq = torch.cat((lay1,lay2),2)

        if self.rnn_layer == 'transformer':
            #for transformer change batch first
            seq = seq.reshape(seq.size(1), seq.size(0), seq.size(2))            
            #OneHotVector = self.E[seq.data.long()].squeeze().to(device)
            EmbeddingVector = self.embedding1(seq.long().to(device)).squeeze().to(device)
            EmbeddingVector = EmbeddingVector.unsqueeze(0).permute(1,0,2)
            #self.rnn_out = self.rnn(EmbeddingVector)                   
            self.rnn_out = self.rnn(EmbeddingVector, EmbeddingVector, tgt_mask=self.rnn.generate_square_subsequent_mask(200).to(device)) 

            #self.rnn_out = self.rnn_out[:,-1,:]  
            self.rnn_out = self.rnn_out.reshape(self.rnn_out.size(1),self.rnn_out.size(0)*self.rnn_out.size(2))            
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

class Encoder():
    def __init__(self, pc_bits=7, concatenate_dir=False):
        self.pc_bits = pc_bits
        self.concatenate_dir = concatenate_dir
    
    def encode(self, pc, direction):
        pc = int(pc)
        pc = (pc & (2**self.pc_bits - 1))<< 1
        if(self.concatenate_dir):
            return pc+int(direction)
        else:
            return pc

    
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
model = None
allBranch = {}

def merger(modelType, trace):
    ### Merge and remove files
    bench = trace.split("_ref")[0]
    benchPath = "../Datasets/myTraces/"+bench+"/"+ trace
    finalPath = "specificBranch/{}/predictionTraces/{}".format(bench, modelType)
    finalFile = gzip.open("{}/{}".format(finalPath, trace), 'wt')
    brs = os.listdir("{}/{}".format(finalPath, trace.split(".champsimtrace")[0]))
    brsDict = {
        int(br[:-3]):
        gzip.open(finalPath+"/"+trace.split(".champsimtrace")[0]+"/"+br, 'rt')
        for br in brs
    }
    with gzip.open(benchPath, 'rt') as fp:
        # write H2Ps first
        H2Ps = str(list(brsDict.keys()))[1:-1].replace(",", "")
        finalFile.write(H2Ps+"\n")
        for line in fp:
            ip = int(line.split(" ")[0])
            if ip in brsDict.keys():
                newline = brsDict[ip].readline()
                _ = finalFile.write(newline)
    for br in brsDict:
        brsDict[br].close()
    for br in brs:
        os.remove("{}/{folderName}/{branchPt}".format(
            finalPath, 
            folderName=trace.split(".champsimtrace")[0], 
            branchPt=br))
    os.rmdir("{}/{}".format(finalPath, trace.split(".champsimtrace")[0]))
    finalFile.close()

def main(outputName, mode, bench, trace, overwrite='False'):    
    
    ## GPU/CPU ##
    device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")


    # Name of pt model
    modelFolder = "./specificBranch/"+bench+"/models/"+mode+"/"
    models = os.listdir(modelFolder)
    modelsDict = {}
    if (mode=="BranchNet"):        
        encoder = Encoder(pc_bits=11,concatenate_dir=True)
        for modelN in models:
            historyLen = 582
            modelName = modelN
            modelName = modelFolder + modelName
            ## Model 
            model = BranchNetModel.BranchNet(yaml.safe_load(open("./branchnet/configs/big.yaml")),BranchNetModel.BranchNetTrainingPhaseKnobs()).to(device)

            try:
                print(modelName)
                checkpoint = torch.load(modelName, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
            except:
                print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
                exit()
            modelsDict[int(modelN[3:-3])] = model
        print(model)
    if (mode=="BranchNetTarsa"):
        encoder = Encoder(pc_bits=7,concatenate_dir=True)
        for modelN in models:
            historyLen = 200
            modelName = modelN
            modelName = modelFolder + modelName
            ## Model 
            model = BranchNetModel.BranchNet(yaml.safe_load(open("./branchnet/configs/tarsa.yaml")),BranchNetModel.BranchNetTrainingPhaseKnobs()).to(device)

            #print(model)
            try:
                #print(modelName)
                checkpoint = torch.load(modelName, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
            except:
                print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
                exit()
            modelsDict[int(modelN[3:-3])] = model
    if (mode=="CNN"):
        encoder = Encoder(pc_bits=7,concatenate_dir=True)
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
            try:
                print(modelName)
                checkpoint = torch.load(modelName, map_location=device)
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
            
            try:
                print(modelName)
                checkpoint = torch.load(modelName)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
            except:
                print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
                exit()
            modelsDict[int(modelN[4:-3])] = model

    if (mode=="Transformer"):
        ## Parameters ##
        n_class = 16
        num_layers = 1
        normalization = False
        input_dim = 32
        rnn_layer = 'transformer'        
        for modelN in models:
            modelName = modelN
            modelName = modelFolder + modelName
            ## Model 
            model = RNNLayer(rnn_layer=rnn_layer, input_dim=input_dim, out_dim=n_class, num_layers=num_layers, normalization=normalization).to(device)

            print(model)
            
            try:
                print(modelName)
                checkpoint = torch.load(modelName, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
            except:
                print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
                exit()
            modelsDict[int(modelN[3:-3])] = model
    
    if ('encodePCList' in checkpoint['input_Data']):
        encodePCList = checkpoint['input_Data']['encodePCList']    
    allBranch={}
    correct = 0
    total=1
    now = time.time()

    if(platform.system()=="Windows"):
        benchPath = "../Datasets/myTraces/"+bench+"/"+ trace
    elif(platform.system()=="Linux"):
        benchPath = "/local/avontz/myTraces/datasets/"+bench+"/"+ trace
    print(benchPath)
    timeRead = 0
    timeEncode = 0
    timePredict = 0   
    timeTotal=0     
    Path("./specificBranch/{bench}/predictionTraces/{mode}/{trace}".format(
        bench=bench,
        mode=mode,
        trace=trace.split(".champsimtrace")[0]
    )).mkdir(parents=True, exist_ok=True)
    
    outputTraceDict = {
        ip:
        gzip.open("./specificBranch/{bench}/predictionTraces/{mode}/{trace}/{PC}.gz".format(
            bench=bench,
            mode=mode,
            trace=trace.split(".champsimtrace")[0]
            PC=str(ip)
            ), 'wt')
        for ip in modelsDict.keys()
    }
    IPsToPredict = list(modelsDict.keys())
    #first line should contain ips predictions     
    # branches that contain "all branches "
    with gzip.open(benchPath, 'rt') as fp:
        try:                
            batchSize = 4096
            batchDict = {
                ip:
                {   'batch':torch.zeros((batchSize,historyLen),dtype=torch.float64,device=device),
                    'labels': [0]*batchSize,
                    'batchCounter':0
                }
                for ip in modelsDict.keys()}                                
            for ip in modelsDict.keys():                    
                    allBranch[ip] = {'total': 1, 'correct' : 0, 'acc': 0}                        
            #ips = [0]*batchSize#torch.zeros((batchSize,1), dtype=torch.float64)
            #labelsDict = {ip: for ip in batchDict.keys()}#torch.zeros((batchSize), dtype=torch.float64)
            #history =  torch.zeros((historyLen), dtype=torch.float64,device=device) 
            historyDq = deque([0]*historyLen)
            #batchCounters = {ip:0 for ip in batchDict.keys()}
            while True:
                timeTotalStart=time.time()
                timeStart = time.time()                                
                line = fp.readline()
                [ipFloat, label] = np.float64(line.split(" "))     
                ip = int(ipFloat)                    
                timeRead += time.time()-timeStart
                #prediction first if ip in dict of branchnets                    
                timeStart = time.time()                                
                if(ip in modelsDict):                
                    total+=1 
                    allBranch[ip]['total']+=1
                    
                    batchCounter = batchDict[ip]['batchCounter']
                    # history = torch.tensor(historyDq).unsqueeze(dim=0)
                    tmp = np.flip(np.array(historyDq)).copy()
                    batchDict[ip]['batch'][batchCounter] = torch.tensor(tmp).unsqueeze(dim=0)#history.detach().clone()                        
                    batchDict[ip]['labels'][batchCounter] = label
                    batchDict[ip]['batchCounter']+=1                                   
                    if(batchDict[ip]['batchCounter'] == batchSize):                             
                        batchDict[ip]['batchCounter']  = 0
                        prediction = 2.0   
                        if(mode=="BranchNetTarsa" or mode=="BranchNet"):
                            prediction = torch.sign(modelsDict[ip](batchDict[ip]['batch'].long())) 
                            #change -1 to 0 by applying relu
                            torch.nn.functional.relu(prediction, inplace=True)
                        elif(mode=="CNN"):
                            prediction = torch.round(modelsDict[ip](batchDict[ip]['batch'].float()))
                        elif (mode=="LSTM"):
                            prediction = (modelsDict[ip](batchDict[ip]['batch'].float()).argmax(axis=1).cpu())
                        elif (mode=='Transformer'):
                            prediction = (modelsDict[ip](batchDict[ip]['batch'].float()).argmax(axis=1).cpu())                               
                        #predictionList = prediction.tolist()                                 
                        # for i in range(batchSize):                        
                        #     if(predictionList[i] == batchDict[ip]['labels'][i]):
                        results = prediction == torch.tensor(batchDict[ip]['labels'], device=device)                            
                        resSum = int(results.sum())    
                        correct+=resSum
                        allBranch[ip]['correct']+=resSum
                        allBranch[ip]['acc'] = allBranch[ip]['correct']/allBranch[ip]['total']                                            
                        predictionList = prediction.tolist()                                                        
                        for pred in predictionList:
                            outputTraceDict[ip].write(str(ip)+" "+str(int(pred))+"\n")      
                        del results    
                        del prediction
                    torch.cuda.empty_cache()                                             
                    
                timeEnd = time.time()
                timePredict += timeEnd - timeStart
                #update history tensor                        
                #encode ip using encoder class
                timeStart = time.time()                       
                encodedPC = encoder.encode(ip, label)
                historyDq.pop()
                historyDq.appendleft(encodedPC)
                timeEnd = time.time()
                timeEncode += timeEnd - timeStart                                                                          
                timeTotal += time.time() - timeTotalStart                                
                if(total%100000==0):                                               
                    ## Calculate remaining batches                        
                    for ip in batchDict:
                        #batchDict[ip]['batch'] = batchDict[ip]['batch'][:batchDict[ip]['batchCounter']-1]
                        #batchDict[ip]['labels'] = batchDict[ip]['labels'][:batchDict[ip]['batchCounter']-1]
                        batchCounter = batchDict[ip]['batchCounter']
                        batchDict[ip]['batchCounter']  = 0
                        prediction = 2.0                                                           
                        if(mode=="BranchNetTarsa" or mode=="BranchNet"):                            
                            prediction = torch.sign(modelsDict[ip](batchDict[ip]['batch'].long()).cpu())
                            torch.nn.functional.relu(prediction, inplace=True)
                        elif(mode=="CNN"):
                            prediction = torch.round(modelsDict[ip](batchDict[ip].float()))
                        elif (mode=="LSTM"):
                            prediction = (modelsDict[ip](batchDict[ip].float()).argmax(axis=1).cpu())
                        elif (mode=='Transformer'):
                            prediction = (modelsDict[ip](batchDict[ip].float()).argmax(axis=1).cpu())                               
                        #predictionList = prediction.tolist()                                 
                        # for i in range(batchDict[ip]['batch'].size(0)):
                        #     if(predictionList[i] == batchDict[ip]['labels'][i]):
                        results = torch.narrow(prediction == torch.tensor(batchDict[ip]['labels']), 0, 0, batchCounter)                
                        resSum = int(results.sum())
                        correct+=resSum
                        allBranch[ip]['correct']+=resSum
                        allBranch[ip]['acc'] = allBranch[ip]['correct']/allBranch[ip]['total']
                        predictionList = torch.narrow(prediction, 0, 0, batchCounter).tolist()
                        del results
                        del prediction
                        for prediction in predictionList:
                            outputTraceDict[ip].write(str(ip)+" "+str(int(prediction))+"\n")  
                                        
                    total+=1
                    print(correct,total, 100*correct/total)
                    p = pprint.PrettyPrinter()
                    p.pprint(allBranch)
                    torch.save(allBranch, outputName)
                    print("Total:", timeTotal, "Read:", timeRead, round((timeRead/timeTotal)*100), "Encode:", timeEncode, round((timeEncode/timeTotal)*100), "Predict:", timePredict, round((timePredict/timeTotal)*100))
            print(correct,total, 100*correct/total)
        except Exception as e:
            for ip in batchDict:
                #batchDict[ip]['batch'] = batchDict[ip]['batch'][:batchDict[ip]['batchCounter']-1]
                #batchDict[ip]['labels'] = batchDict[ip]['labels'][:batchDict[ip]['batchCounter']-1]
                batchCounter = batchDict[ip]['batchCounter']
                batchDict[ip]['batchCounter']  = 0
                prediction = 2.0                                                           
                if(mode=="BranchNetTarsa" or mode=="BranchNet"):                            
                    prediction = torch.sign(modelsDict[ip](batchDict[ip]['batch'].long()).cpu())
                    torch.nn.functional.relu(prediction, inplace=True)
                elif(mode=="CNN"):
                    prediction = torch.round(modelsDict[ip](batchDict[ip].float()))
                elif (mode=="LSTM"):
                    prediction = (modelsDict[ip](batchDict[ip].float()).argmax(axis=1).cpu())
                elif (mode=='Transformer'):
                    prediction = (modelsDict[ip](batchDict[ip].float()).argmax(axis=1).cpu())                               
                #predictionList = prediction.tolist()                                 
                # for i in range(batchDict[ip]['batch'].size(0)):
                #     if(predictionList[i] == batchDict[ip]['labels'][i]):
                results = torch.narrow(prediction == torch.tensor(batchDict[ip]['labels']), 0, 0, batchCounter)                
                resSum = int(results.sum())
                correct+=resSum
                allBranch[ip]['correct']+=resSum
                allBranch[ip]['acc'] = allBranch[ip]['correct']/allBranch[ip]['total']
                predictionList = torch.narrow(prediction, 0, 0, batchCounter).tolist()
                del results
                del prediction
                for prediction in predictionList:
                    outputTraceDict[ip].write(str(ip)+" "+str(int(prediction))+"\n")  
            print("ERROR" ,e)
            print(correct,total, 100*correct/total)
            p.pprint(allBranch)
            torch.save(allBranch, outputName)
    end = time.time()
    print("total time taken to check: ", end-now)        
    # print("merging results")
    
if __name__ == '__main__':
    if(len(sys.argv) < 5):
        print("usage: script 'outputName', MODE:LSTM/CNN, benchmark, trace, overwrite=true/false(default false)")
        exit()
    # check to overwite output (default no)    
    overwrite='True'
    if (len(sys.argv) == 6):
        overwrite = sys.argv[5]
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], overwrite=overwrite)