## Test CNN regression Models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.utils.data import DataLoader
import time
import readRegression
from readRegression import BranchDataset
import os


class TwoLayerFullPrecisionBPCNN(nn.Module):
    def __init__(self, tableSize=256, numFilters=2, historyLen=200):
        super().__init__()
        
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
        h1a = h1a.reshape(len(h1a),historyLen*numFilters)
        out = self.l2(h1a)        
        out = self.sigmoid2(out)
        return out

## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")

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
paramsValid = {'batch_size': 5000,
          'shuffle': False,
          'num_workers': 2}

# Benchmark
input_bench = ["600.perlbench_s-210B.champsimtrace.xz._.dataset_unique.txt.gz"]
startSample, endSample = 100, 400
encodePCList=True
loadPt=True

try:
    print(modelName)
    checkpoint = torch.load(modelName)
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
    exit()

## Optimize 
criterion = nn.L1Loss()


print("Loading ValidationDataset")
if(loadPt):
    valid = torch.load("../Datasets/train_600_210B_600K.pt")
else:
    _, valid = read.readFileList(input_bench, startSample,endSample, ratio=0.0)
    #torch.save(valid, '600.perlbench_1273B_valid600K.pt')

validation_set = BranchDataset(valid, encodePCList=encodePCList)

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

    correct += (torch.round(outputs.cpu()) == Validlabels.cpu()).sum()
epochLoss = float(sum(loss_values))/float(len(valid_loader))
epochAccuracy = float(correct)/float(len(validation_set))        
print("validation: loss:", epochLoss ,"acc:" , epochAccuracy)    
print("Finish")

