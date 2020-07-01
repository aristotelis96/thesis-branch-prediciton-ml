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


class TwoLayerFullPrecisionBPCNN(nn.Module):
    def __init__(self, tableSize=256, numFilters=2, historyLen=200):
        super().__init__()
        
        #embed via identity matrix:
        self.E = torch.eye(tableSize, dtype=torch.float32)
        
        #convolution layer, 1D in effect; Linear Layer
        self.c1 = nn.Conv2d(16, numFilters, (1,1))
        self.tahn = nn.Tahn()
        self.l2 = nn.Linear(historyLen*numFilters, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq):
        #Embed by expanding sequence of ((IP << 7) + dir ) & (255)
        # integers into 1-hot history matrix during training
        
        xFull = self.E[seq.data]
        h1 = self.c1(xFull)
        h1a = self.tahn(h1)
        h2 = self.l2(h1a)
        out = self.sigmoid(h2)
        return out

## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")

## Parameters ##
tableSize = 256
numFilters = 2
historyLen = 200

# learning Rate
learning_rate = 1e-3

# Load a checkpoint?
ContinueFromCheckpoint = False

# Epoch
epochStart = 0
epochEnd = 50

## Model 
model = TwoLayerFullPrecisionBPCNN(tableSize=tableSize, numFilters=numFilters, historyLen=historyLen).to(device)

print(model)
## TRAIN/TEST DATALOADER
# Parameters
paramsTrain = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 4}
paramsValid = {'batch_size': 10000,
          'shuffle': False,
          'num_workers': 4}

# Benchmark
input_bench = ["600.perlbench_s-1273B.champsimtrace.xz._.dataset_unique.txt.gz"]
startSample, endSample = 100, 100000
inputDescription = '1-hot sequence-200 only Taken/NotTaken, no program counter'

print("Loading TrainDataset")
print("Loading ValidationDataset")
train, valid = read.readFileList(input_bench, startSample,endSample)

training_set, validation_set = BranchDataset(train), BranchDataset(valid)

train_loader = DataLoader(training_set, **paramsTrain)
valid_loader = DataLoader(validation_set, **paramsValid)

##
## Optimize 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate/10)

total_Train_loss = []
total_Train_accuracy = []
total_Validation_loss = []
total_Validation_accuracy = []


if ContinueFromCheckpoint:
    try:
        checkpoint = torch.load("checkpointCNN.pt")
        epochStart = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_Train_loss = checkpoint['total_Train_loss']
        total_Train_accuracy = checkpoint['total_Train_accuracy']
        total_Validation_loss = checkpoint['total_Validation_loss']
        total_Validation_accuracy = checkpoint['total_Validation_accuracy']
    except:
        print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
        print("STARTING TRAINING WITHOUT LOADING FROM CHECKPOINT FROM EPOCH 0")        


#TRAINING
now = time.time()
epoch = epochStart
while epoch < epochEnd:
    try:
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
        ## SAVE A CHECKPOINT as "checkpointCNN.pt"
        torch.save({
                'epoch': epoch,
				'learning_rate': learning_rate,
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
                    'type': inputDescription,
                    'batch_size': paramsTrain['batch_size']
                },
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_Validation_accuracy': total_Validation_accuracy,
                'total_Validation_loss': total_Validation_loss,
                'total_Train_loss': total_Train_loss,
                'total_Train_accuracy': total_Train_accuracy
            }, "checkpointCNN.pt")
    except Exception as e:
        print("Error occured, reloading model from previous iteration and skipping epoch increase")
        print(e)
        checkpoint = torch.load("checkpointCNN.pt")
        epochStart = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_Train_loss = checkpoint['total_Train_loss']
        total_Train_accuracy = checkpoint['total_Train_accuracy']
        total_Validation_loss = checkpoint['total_Validation_loss']
        total_Validation_accuracy = checkpoint['total_Validation_accuracy']        
        continue
    epoch+=1 
    
end = time.time()
print("total time taken to train: ", end-now)

print("Finish")
