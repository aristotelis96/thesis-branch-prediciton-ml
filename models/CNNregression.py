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
import sys
from math import ceil, floor
from branchnet import model as BranchNetModel
import yaml


class TwoLayerFullPrecisionBPCNN(nn.Module):
    def __init__(self, tableSize=256, numFilters=2, historyLen=200):
        super().__init__()
        
        #embed via identity matrix:
        self.E = torch.eye(tableSize, dtype=torch.float32).to(device)
        
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
        #xFull = xFull.reshape(len(xFull),16,16,200).to(device)
        
        h1 = self.c1(xFull)
        h1a = self.tahn(h1)        
        h1a = h1a.reshape(h1a.size(0),h1a.size(1)*h1a.size(3))
        out = self.l2(h1a)        
        out = self.sigmoid2(out)
        return out

## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def main(branch, benchmark="ERROR"):
    ## Parameters ##
    tableSize = 256
    numFilters = 2
    historyLen = 200

    # learning Rate
    learning_rate = 2e-3

    # Load a checkpoint?
    ContinueFromCheckpoint = False

    # Epoch
    epochStart = 0
    epochEnd = 20

    ## Model 
    #model = TwoLayerFullPrecisionBPCNN(tableSize=tableSize, numFilters=numFilters, historyLen=historyLen).to(device)
    model = BranchNetModel.BranchNet(yaml.safe_load(open("./branchnet/configs/tarsa.yaml")),BranchNetModel.BranchNetTrainingPhaseKnobs()).to(device)
    pc_bits = 7
    hash_dir = True


    print(model)
    ## TRAIN/TEST DATALOADER
    # Parameters
    paramsTrain = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 2}
    paramsValid = {'batch_size': 4096,
            'shuffle': False,
            'num_workers': 2}

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


    total = torch.load("./specificBranch/"+benchmark+"/datasets/"+str(branch)+".pt")        
    
    totalDataset = BranchDataset(total, encodePCList=encodePCList, historyLen=historyLen, pc_bits=pc_bits, hash_dir=hash_dir)
    #split into train and valid set using ratio 
    splitRatio = [0.95, 0.05]
    training_set, validation_set = torch.utils.data.random_split(totalDataset, [ceil(len(total)*splitRatio[0]), floor(len(total)*splitRatio[1])])#BranchDataset(train, encodePCList=encodePCList), BranchDataset(valid, encodePCList=encodePCList)    
    #create dataloaders
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
                outputs = model(X.long())
                
                loss = criterion(outputs, labels.float())                 
                

                loss.backward()

                optimizer.step()
                # print statistics
                running_loss += loss.item()
                correct += float((torch.sign(outputs.cpu()) == labels.cpu()).sum())
            train_acc = correct/float(len(training_set))
            train_loss = running_loss/float(len(train_loader))
            total_Train_accuracy.append(train_acc)
            total_Train_loss.append(train_loss)
            print("Epoch: ", epoch, "train loss:", train_loss, " acc:", train_acc)
            if(correct/float(len(training_set))>0.99):
                break
            correct = 0
            for X_val, Validlabels in valid_loader:
                model.eval() 

                X_val = X_val.to(device)
                Validlabels = Validlabels.to(device)

                outputs = model(X_val.long())

                loss = criterion(outputs, Validlabels.float())

                loss_values.append(loss.item())    
            
                correct += (torch.sign(outputs.cpu()) == Validlabels.cpu()).sum()
            epochLoss = float(sum(loss_values))/float(len(valid_loader))
            total_Validation_loss.append(epochLoss)

            epochAccuracy = float(correct)/float(len(validation_set))
            total_Validation_accuracy.append(epochAccuracy)
            print("Epoch: ", epoch, "validation: loss:", epochLoss ,"acc:" , epochAccuracy)
            # if(epochAccuracy>0.99):
            #     break
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
                }, "./specificBranch/"+benchmark+"/models/BranchNet/CNN"+str(branch)+".pt")
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

if __name__ == "__main__":
    benchmark=sys.argv[1]
    branches = os.listdir('./specificBranch/'+benchmark+'/datasets')
    branches = [int(x[:-3]) for x in branches]
    oldbranches = os.listdir("./specificBranch/"+benchmark+"/models/BranchNet")
    # for branch in oldbranches:
    #    if int(branch[3:-3]) in branches: branches.remove(int(branch[3:-3]))            
    for branch in branches:
        print("Now training for branch:", branch)
        main(branch, benchmark=benchmark)

