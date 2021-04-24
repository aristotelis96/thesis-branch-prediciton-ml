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
from branchnet import dataset_loader as BranchNetDataset
import yaml


class TwoLayerFullPrecisionBPCNN(nn.Module):
    def __init__(self, tableSize=256, numFilters=2, history_length=200):
        super().__init__()
        
        #embed via identity matrix:
        self.E = torch.eye(tableSize, dtype=torch.float32).to(device)
        
        #convolution layer, 1D in effect; Linear Layer
        self.c1 = nn.Conv2d(tableSize, numFilters, (1,1))        
        self.tahn = nn.Tanh()
        self.l2 = nn.Linear(history_length*numFilters, 1)        
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

""" Trains a model on a branch using h5py Datasets """
def Trainer(branch_pc, benchmark="ERROR", mode="BranchNet"):
    ## Parameters ##
    tableSize = 256
    numFilters = 2
    history_length = 582

    # learning Rate
    learning_rate = 2e-3

    Training_steps = [1000, 1000, 1000] 
    Validation_steps = [1000]

    ## Model 
    mode = mode
    #model = TwoLayerFullPrecisionBPCNN(tableSize=tableSize, numFilters=numFilters, history_length=history_length).to(device)
    model = BranchNetModel.BranchNet(yaml.safe_load(open("./branchnet/configs/big.yaml")),BranchNetModel.BranchNetTrainingPhaseKnobs()).to(device)
    pc_bits = 11
    pc_hash_bits = 11
    hash_dir = False


    print(model)
    
    ## TRAIN/TEST DATALOADER
    # Parameters
    paramsTrain = {'batch_size': 2048,
            'shuffle': True,
            'num_workers': 0}
    paramsValid = {'batch_size': 512,
            'shuffle': True,
            'num_workers': 0}

    trace_folder = "C:/Users/Aristotelis/Desktop/diploma/Datasets/h5pyDatasets/{}".format(benchmark)
    trace_paths = [
        "{}/{}".format(trace_folder, traceFile) for traceFile in os.listdir(trace_folder)
    ]
    dataset = BranchNetDataset.BranchDataset(
        trace_paths=trace_paths, 
        br_pc=branch_pc, 
        history_length=history_length,
        pc_bits = pc_bits,
        pc_hash_bits = pc_hash_bits,
        hash_dir_with_pc = hash_dir,
        in_mem=False)
    
    #create dataloaders
    train_loader = DataLoader(dataset, **paramsTrain)
    valid_loader = DataLoader(dataset, **paramsValid)

    ## Optimize 
    criterion = nn.MSELoss()    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate/10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    total_Train_loss = []
    total_Train_accuracy = []
    total_Validation_loss = []
    total_Validation_accuracy = []

    #TRAINING
    now = time.time()
    # try:             
    #return dataset
    for num_steps in Training_steps:
        print('Training for {} steps with learning rate {}'.format(
        num_steps, scheduler.get_last_lr()[0]))       
        step = 0
        while step < num_steps:                        
            running_loss = 0.0
            correct = 0.0
            for X, labels in train_loader:
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
                # step increase
                step += 1
                if (step >= num_steps):
                    break 
            scheduler.step()
     
        train_acc = correct/float(num_steps*train_loader.batch_size)
        train_loss = running_loss/float(num_steps*train_loader.batch_size)
        total_Train_accuracy.append(train_acc)
        total_Train_loss.append(train_loss)
        print("Step: {} --- Accuracy: {}% --- Loss: {}".format(
            step, 
            round(train_acc*100, 4),
            train_loss
        ))
    for num_steps in Validation_steps:
        print('Validate net for {} steps with random samples'.format(
        num_steps))
        step=0       
        while step < num_steps:
            running_loss = 0.0
            correct = 0.0
            for X_val, Validlabels in valid_loader:
                model.eval() 

                X_val = X_val.to(device)
                Validlabels = Validlabels.to(device)

                outputs = model(X_val.long())

                loss = criterion(outputs, Validlabels.float())

                running_loss+=(loss.item())    
            
                correct += (torch.sign(outputs.cpu()) == Validlabels.cpu()).sum()
                # Step increase
                step+=1
                if step >= num_steps:
                    break
        validation_loss = float(running_loss)/float(num_steps*valid_loader.batch_size)
        total_Validation_loss.append(validation_loss)

        validation_accuracy = float(correct)/float(num_steps*valid_loader.batch_size)
        total_Validation_accuracy.append(validation_accuracy)
        print("Evaluation of model with random samples. Accuracy: {}% --- Loss: {}".format(
            round(validation_accuracy*100, 4),
            validation_loss
        ))    
    ## SAVE A CHECKPOINT as "checkpointCNN.pt"
    torch.save({
            'learning_rate': learning_rate,
            'model_architecture': str(model),                
            'input_Data': {
                'training_steps' : Training_steps
            },
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_Validation_accuracy': total_Validation_accuracy,
            'total_Validation_loss': total_Validation_loss,
            'total_Train_loss': total_Train_loss,
            'total_Train_accuracy': total_Train_accuracy
        }, "./specificBranch/"+benchmark+"/models/"+mode+"/CNN"+str(branch)+".pt")

    end = time.time()
    print("total time taken to train: ", end-now)

    print("Finish")

if __name__ == "__main__":
    benchmark=sys.argv[1]
    mode = sys.argv[2]
    branches = os.listdir('./specificBranch/'+benchmark+'/datasets')
    branches = [int(x[:-3]) for x in branches]
    # oldbranches = os.listdir("./specificBranch/"+benchmark+"/models/BranchNet")
    # for branch in oldbranches:
    #     if int(branch[3:-3]) in branches: branches.remove(int(branch[3:-3]))            
    for branch in branches:
        print("Now training for branch:", branch)
        Trainer(branch, benchmark=benchmark, mode=mode)

