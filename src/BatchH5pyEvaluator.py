import re
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.utils.data import DataLoader
import time
import read
import os
import numpy as np
import gzip
import pprint
import branchnet.model as BranchNetModel
import yaml
from pathlib import Path
import gc
from collections import deque
from models import TwoLayerFullPrecisionBPCNN, RNNLayer, Encoder
from branchnet import dataset_loader as BranchNetDataset

device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
model = None
allBranch = {}

def merger(modelType, trace):
    ### Merge and remove files
    bench = trace.split("_ref")[0]
    benchPath = "../Datasets/myTraces/"+bench+"/"+ trace
    finalPath = "outputs/{}/predictionTraces/{}".format(bench, modelType)
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

def main(outputName, mode, benchmark, TracePath):    
    
    ## GPU/CPU ##
    device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")


    # Name of pt model
    modelFolder = "./outputs/"+benchmark+"/models/"+mode+"/"
    models = os.listdir(modelFolder)
    modelsDict = {}
    if (mode=="BranchNetLSTM"):        
        encoder = Encoder(pc_bits=11,concatenate_dir=True)
        for modelN in models:
            historyLen = 582
            modelName = modelN
            modelName = modelFolder + modelName
            ## Model 
            model = BranchNetModel.BranchNet(yaml.safe_load(open("./branchnet/configs/BranchNetLSTM.yaml")),BranchNetModel.BranchNetTrainingPhaseKnobs()).to(device)

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
    if (mode=="BranchNet"):                
        pc_bits = 11
        pc_hash_bits = 11
        hash_dir = False
        ProblemType = "Regression"
        history_length = 582
        for modelN in models:                    
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
            model = TwoLayerFullPrecisionBPCNN(tableSize=tableSize, numFilters=numFilters, history_length=historyLen).to(device)

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
        encoder = Encoder(pc_bits=11,concatenate_dir=True)
        ProblemType = "Regression"
        history_length = 582
        ## Parameters ##
        n_class = 128
        num_layers = 1
        normalization = False
        input_dim = 32
        rnn_layer = 'lstm'
        pc_bits = 11
        pc_hash_bits = 11
        hash_dir = False

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
            modelsDict[int(modelN[3:-3])] = model

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
            model = RNNLayer(rnn_layer=rnn_layer, input_dim=input_dim, out_dim=n_class, num_layers=num_layers, normalization=normalization).cpu()

            print(model)
            
            try:
                print(modelName)
                checkpoint = torch.load(modelName, map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
            except:
                print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
                exit()
            modelsDict[int(modelN[3:-3])] = model
    
    # Create Stat result folder
    Path(os.path.dirname(outputName)).mkdir(parents=True, exist_ok=True)
    # Create Predictions Dir
    SimpointFolderName = os.path.basename(TracePath).split("_dataset")[0]
    Path("./outputs/{bench}/predictionTraces/{mode}/{SimpointFolderName}".format(
        bench=bench,
        mode=mode,
        SimpointFolderName=SimpointFolderName
    )).mkdir(parents=True, exist_ok=True)
    
    H5pyFolder = "C:/Users/Aristotelis/Desktop/diploma/Datasets/h5pyDatasetsEvaluation/{}".format(benchmark)
    H5pyPaths = [
        "{}/{}".format(H5pyFolder, traceFile) for traceFile in os.listdir(H5pyFolder)
    ]
    
    for trace_file in H5pyPaths:
        BranchesStats = {}
        for branch in modelsDict.keys():   
            BranchesStats[branch] = {}            
            outputPredictionTrace = gzip.open("./outputs/{bench}/predictionTraces/{mode}/{SimpointFolderName}/{PC}.gz".format(
                bench=bench,
                mode=mode,
                SimpointFolderName=SimpointFolderName,
                PC=str(branch)
                ), 'wt')            
            model = modelsDict[branch].to(device)
            
            paramsValid = {'batch_size': 4096,
                'shuffle': False,
                'num_workers': 0}

            
            
            dataset = BranchNetDataset.BranchDataset(
                trace_paths=[trace_file], 
                br_pc=branch, 
                history_length=history_length,
                pc_bits = pc_bits,
                pc_hash_bits = pc_hash_bits,
                hash_dir_with_pc = hash_dir,
                in_mem=False)
            eval_loader = DataLoader(dataset, **paramsValid)
            total = 0
            correct = 0
            BranchesStats[branch]['total'] = 0
            BranchesStats[branch]['correct'] = 0
            print("Will now evalute branch: {} benchmark: {} h5pyDataset: {}".format(
                branch, benchmark, trace_file
            ))
            for X, labels in eval_loader:                
                X = X.to(device)
                
                outputs = model(X.long())  
                
                # print statistics                
                if ProblemType=="Classification":
                    correct += float((outputs.argmax(axis=1).cpu() == labels.cpu()).sum())
                elif ProblemType == "Regression":

                    correct += float((torch.sign(outputs.cpu()) == labels.cpu()).sum())
                                
                total += eval_loader.batch_size                
                BranchesStats[branch]['total'] += total
                BranchesStats[branch]['correct'] += correct           
                if (total / eval_loader.batch_size) % 10 == 0:
                    train_acc = correct/float(total)
                    print("Accuracy: {}% - Correct: {} - Total: {}".format(                        
                        train_acc*100,
                        correct,
                        total
                    ))
                # modify outputs to 0 and 1 for prediction Log Trace
                outputs = torch.sign(outputs)
                F.relu(outputs, inplace=True)
                for output in outputs:
                    outputPredictionTrace.write(str(branch)+" "+str(int(output))+"\n")      
                del outputs
                torch.cuda.empty_cache()  
            BranchesStats[branch]['acc'] = float(correct)/float(total)
        torch.save(BranchesStats, outputName)
        outputPredictionTrace.close()
        traceName = trace_file.split("_dataset")[0]+".champsimtrace.xz._.allBranches.txt.gz"        
        merger(mode, traceName) 
                                  
                
if __name__ == '__main__':
    if(len(sys.argv) < 3):
        print("Python program to evaluate a NeuralNetwork across all simpoints of Benchmark")
        print("usage: script NN:LSTM/CNN, benchmark")
        exit()
    # check to overwite output (default no)    
    NeuralNetwork = sys.argv[1]
    bench = sys.argv[2]
    outputFolder = "./outputs/{}/resultStats/{}".format(bench, NeuralNetwork)

    if(platform.system()=="Windows"):
        H5pyFolder = "C:/Users/Aristotelis/Desktop/diploma/Datasets/h5pyDatasetsEvaluation/{}".format(bench)
    elif(platform.system()=="Linux"):
        H5pyFolder = "/local/avontz/myTraces/datasets/{}".format(bench)
        print("NEED TO SET UP LINUX VERSION FOLDER")
        assert 1
    
    H5pyPaths = [
        "{}/{}".format(H5pyFolder, h5py) for h5py in os.listdir(H5pyFolder)
        ]
    # skip existing    
    # existing = os.listdir(outputFolder)
    # CHECK IF EXISTS AND SKIP, DELETE LINE ABOVE    
    for H5pyPath in H5pyPaths:
        simpoint = re.search("-([0-9]*?)B.", H5pyPath).group(1) # extract simpoint
        output = "{outputFolder}/{bench}-{simpoint}B.pt".format(
            outputFolder = outputFolder,
            bench = bench,
            simpoint = simpoint)          
        # SPECIAL: CHECK IF OUTPUT ALREADY EXISTS AND SKIP
        # if os.path.basename(output) in existing: continue 
        # CHECK IF EXISTS AND SKIP, DELETE LINE ABOVE
        print("Now will run Benchmark: {}, Simpoint: {}B, with NeuralN: {}".format(
            bench, simpoint, NeuralNetwork
        ))              
        main(output, NeuralNetwork, bench, H5pyPath)    
    exit()
