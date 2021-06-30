import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.utils.data import DataLoader
import time
import os
import sys
from math import ceil, floor
from branchnet import model as BranchNetModel
from branchnet import dataset_loader as BranchNetDataset
import yaml
import models 
import numpy as np
## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

""" Trains a model on a branch using h5py Datasets """
def Trainer(branch_pc, benchmark="ERROR", mode="BranchNet"):    
    # learning Rate
    learning_rate = 1e-3

    Training_steps = [2000, 1000, 1000] 
    Validation_steps = [100]

    ## Model 
    mode = mode
    if "BranchNetTransformer" == mode:
        ProblemType = "Regression"
        history_length = 582
        model = BranchNetModel.BranchNet(yaml.safe_load(open("./branchnet/configs/bigTransformer.yaml")),BranchNetModel.BranchNetTrainingPhaseKnobs()).to(device)
        pc_bits = 11
        pc_hash_bits = 11
        hash_dir = False
    if "CNN" == mode:
        ProblemType = "Regression"
        history_length = 200
        tableSize = 256
        numFilters = 2
        model = models.TwoLayerFullPrecisionBPCNN(tableSize=tableSize, numFilters=numFilters, history_length=history_length).to(device)    
        pc_bits = 7
        pc_hash_bits = 7
        hash_dir = False
    if "BranchNetLSTM" == mode:
        ProblemType = "Regression"
        history_length=582
        model = BranchNetModel.BranchNet(yaml.safe_load(open("./branchnet/configs/BranchNetLSTM.yaml")),BranchNetModel.BranchNetTrainingPhaseKnobs()).to(device)
        pc_bits = 11
        pc_hash_bits = 11
        hash_dir = False

    if "BranchNet" == mode:
        ProblemType = "Regression"
        history_length = 582
        model = BranchNetModel.BranchNet(yaml.safe_load(open("./branchnet/configs/big.yaml")),BranchNetModel.BranchNetTrainingPhaseKnobs()).to(device)
        pc_bits = 11
        pc_hash_bits = 11
        hash_dir = False
    if "BranchNetTarsa" == mode:
        ProblemType = "Regression"
        history_length = 200
        model = BranchNetModel.BranchNet(yaml.safe_load(open("./branchnet/configs/tarsa.yaml")),BranchNetModel.BranchNetTrainingPhaseKnobs()).to(device)
        pc_bits = 7
        pc_hash_bits = 7
        hash_dir = False
    if mode =="Transformer":
        ProblemType = "Classification"
        history_length = 200
        pc_bits = 11
        pc_hash_bits = 11
        hash_dir = False
        n_class = 32
        num_layers = 1
        normalization = False
        input_dim = 32
        rnn_layer = 'transformer'
        model = models.RNNLayer(rnn_layer=rnn_layer, input_dim=input_dim, out_dim=n_class, num_layers=num_layers, pc_hash_bits=pc_hash_bits, normalization=normalization).to(device)
    if mode=="LSTM":
        ProblemType = "Regression"
        history_length = 582
        pc_bits = 11
        pc_hash_bits = 11
        hash_dir = False
        n_class = 128
        num_layers = 1
        normalization = False
        input_dim = 32
        rnn_layer = 'lstm'
        model = models.RNNLayer(rnn_layer=rnn_layer, input_dim=input_dim, out_dim=n_class, num_layers=num_layers, normalization=normalization).to(device)
    print(model)
    
    ## TRAIN/TEST DATALOADER
    # Parameters
    paramsTrain = {'batch_size': 128,
            'shuffle': True,
            'num_workers': 0}
    paramsValid = {'batch_size': 128,
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
    ## If transformer or lstm you need classification criterion
    if ProblemType =="Classification":
        # weighted classes
        pos_weight = dataset.get_pos_weight()
        weightClass = torch.tensor([pos_weight, 1/pos_weight], dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=weightClass)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=learning_rate/10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.5, gamma=0.2)

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
                if ProblemType=="Classification":
                    nn.functional.relu(labels, inplace=True)
                    loss = criterion(outputs, labels.long())                 
                else:
                    loss = criterion(outputs, labels.float())                 
                

                loss.backward()

                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if ProblemType=="Classification":
                    correct += float((outputs.argmax(axis=1).cpu() == labels.cpu()).sum())
                else:
                    correct += float((torch.sign(outputs.cpu()) == labels.cpu()).sum())
                # step increase
                step += 1
                if (step >= num_steps):
                    break 
                if step % 100==0:
                    train_acc = correct/float(step*train_loader.batch_size)
                    train_loss = running_loss/float(step*train_loader.batch_size)
                    print("Timestep: {} - Accuracy: {}% - Loss: {}".format(
                        step,
                        train_acc*100,
                        train_loss
                    ))
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

                if ProblemType=="Classification":
                    nn.functional.relu(Validlabels, inplace=True)
                    loss = criterion(outputs, Validlabels.long())                 
                else:
                    loss = criterion(outputs, Validlabels.float())                 
                
                running_loss+=(loss.item())    
            
                if ProblemType=="Classification":
                    correct += float((outputs.argmax(axis=1).cpu() == Validlabels.cpu()).sum())
                else:
                    correct += float((torch.sign(outputs.cpu()) == Validlabels.cpu()).sum())
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
        }, "./outputs/"+benchmark+"/models/"+mode+"/CNN"+str(branch)+".pt")

    end = time.time()
    print("total time taken to train: ", end-now)

    print("Finish")

if __name__ == "__main__":
    if(len(sys.argv)<3):
        print("Trainer for a model and a benchmark.")
        print("Usage ./script model benchmark")
        exit()
    mode = sys.argv[1]
    benchmark=sys.argv[2]
    branches = np.sort(np.load("outputs/{}/H2Ps{}.npy".format(benchmark, benchmark)))
    outputPath = "./outputs/"+benchmark+"/models/"+mode
    if not os.path.exists(outputPath): os.makedirs(outputPath)
    oldbranches = [int(br[3:-3]) for br in os.listdir("./outputs/"+benchmark+"/models/"+mode)]
    # Skip already created
    for branch in oldbranches:
            branches = np.delete(branches, np.where(branches == branch))
    for branch in branches:
        print("Now training for branch:", branch)
        Trainer(branch, benchmark=benchmark, mode=mode)

