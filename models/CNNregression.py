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
        
        #xFull = self.E[seq.data.type(torch.long)]
        xFull = self.E[seq.data.long()]
        xFull = torch.unsqueeze(xFull, 1)        
        xFull = xFull.permute(0,3,1,2)#.to(device)
        #xFull = xFull.reshape(len(xFull),32,8,200).to(device)
        
        h1 = self.c1(xFull)
        h1a = self.tahn(h1)        
        h1a = h1a.reshape(h1a.size(0),h1a.size(1)*h1a.size(3))
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

# learning Rate
learning_rate = 1e-4

# Load a checkpoint?
ContinueFromCheckpoint = True

# Epoch
epochStart = 0
epochEnd = 0

## Model 
model = TwoLayerFullPrecisionBPCNN(tableSize=tableSize, numFilters=numFilters, historyLen=historyLen).to(device)

print(model)
## TRAIN/TEST DATALOADER
# Parameters
paramsTrain = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 2}
paramsValid = {'batch_size': 5000,
          'shuffle': False,
          'num_workers': 2}

# Benchmark
input_bench = ["600.perlbench_s-210B.champsimtrace.xz._.dataset_random.txt.gz"]
startSample, endSample = 100, 400
ratio = 0.75
encodePCList=True
loadPt=True
inputDescription = '1hot matrix, 256*200. 1 at the position of the encoded PC with Taken/NotTaken or 0 otherwise'



##
## Optimize 
criterion = nn.MSELoss()
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


print("Loading TrainDataset")
print("Loading ValidationDataset")
if(loadPt):
    train, valid = torch.load("./../Datasets/train_600_210B_600K.pt"), torch.load("./../Datasets/valid_600_210B_600K-800K.pt")    
    train = torch.load("./600_210B_random_HighOccurency.pt")
    #valid = torch.load("./631_5000.pt")  
    valid =  torch.chunk(valid, 20, dim=0)[0]
    #valid = torch.load("./NewValidation.pt")
else:
    train, valid = readRegression.readFileList(input_bench, startSample,endSample, ratio=ratio)
    #torch.save(train, "train_600_210B_600K")
    #torch.save(valid, "valid_600_210B_200K")

training_set, validation_set = BranchDataset(train, encodePCList=encodePCList), BranchDataset(valid, encodePCList=encodePCList)

train_loader = DataLoader(training_set, **paramsTrain)
valid_loader = DataLoader(validation_set, **paramsValid)


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

            loss = criterion(outputs, labels.float()) 

            

            loss.backward()

            optimizer.step()
            # print statistics
            running_loss += loss.item()
            correct += float((torch.round(outputs.cpu()) == labels.cpu()).sum())
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
        
            correct += (torch.round(outputs.cpu()) == Validlabels.cpu()).sum()
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

