import re
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.utils.data import DataLoader
import time
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
from math import floor

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

def main(outputName, mode, bench, TracePath):    
    ## GPU/CPU ##
    device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
    torch.no_grad()

    # Name of pt model
    modelFolder = "./outputs/"+bench+"/models/"+mode+"/"
    models = os.listdir(modelFolder)
    modelsDict = {}
    if (mode=="BranchNetLSTM"):        
        ProblemType = "Regression"
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
        ProblemType = "Regression" 
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
        ProblemType = "Regression"
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
        ProblemType = "Regression"
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
        ProblemType = "Regression"        
        encoder = Encoder(pc_bits=11,concatenate_dir=True)
        historyLen = 582
        ## Parameters ##
        n_class = 128
        num_layers = 1
        normalization = False
        input_dim = 32
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
            modelsDict[int(modelN[3:-3])] = model

    if (mode=="Transformer"):
        ProblemType = "Classification"
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
    
    # Find maximum Batch size
    max = 2**16
    min = 2
    median = (max-min)/2 + min
    while (max-min)>100:
        try:
            median = (max-min)/2 + min
            print(max,min,median)
            x = torch.randint(2**11,(int(median),582), device=torch.device('cpu'))
            out = modelsDict[list(modelsDict.keys())[0]](x.to(device)        )
            min = median
            del x
            del out
            continue
        except Exception as err:
            print(err)
            max = median
    batchSize = floor(median)
    print("Maximum possible batch size:", median)

    allBranch={}
    correct = 0
    total=1
    now = time.time()
    
    timeRead = 0
    timeEncode = 0
    timePredict = 0   
    timeTotal=0     
    # Create Result Dir
    Path(os.path.dirname(outputName)).mkdir(parents=True, exist_ok=True)
    # Create Predictions Dir
    trace = os.path.basename(TracePath)
    Path("./outputs/{bench}/predictionTraces/{mode}/{trace}".format(
        bench=bench,
        mode=mode,
        trace=trace.split(".champsimtrace")[0]
    )).mkdir(parents=True, exist_ok=True)
    
    outputTraceDict = {
        ip:
        gzip.open("./outputs/{bench}/predictionTraces/{mode}/{trace}/{PC}.gz".format(
            bench=bench,
            mode=mode,
            trace=trace.split(".champsimtrace")[0],
            PC=str(ip)
            ), 'wt')
        for ip in modelsDict.keys()
    }
    
    # First line should contain ips predictions     
    # branches that contain "all branches "
    with gzip.open(TracePath, 'rt') as fp:
        try:                
            batchSize = batchSize
            batchDict = {
                ip:
                {   'batch':torch.zeros((batchSize,historyLen),dtype=torch.float64,device=torch.device('cpu')),
                    'labels': [0]*batchSize,
                    'batchCounter':0
                }
                for ip in modelsDict.keys()}                                
            for ip in modelsDict.keys():                    
                    allBranch[ip] = {'total': 1, 'correct' : 0, 'acc': 0}            
            historyDq = deque([0]*historyLen)            
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
                    tmp = np.flip(np.array(historyDq)).copy()
                    batchDict[ip]['batch'][batchCounter] = torch.tensor(tmp, device=torch.device('cpu')).unsqueeze(dim=0)#history.detach().clone()                        
                    batchDict[ip]['labels'][batchCounter] = label
                    batchDict[ip]['batchCounter']+=1                                   
                    if(batchDict[ip]['batchCounter'] == batchSize):                             
                        batchDict[ip]['batchCounter']  = 0
                        prediction = 2.0   
                        
                        if ProblemType == "Regression":
                            prediction = torch.sign(modelsDict[ip](batchDict[ip]['batch'].long().to(device)))
                            #change -1 to 0 by applying relu
                            F.relu(prediction, inplace=True)
                        elif ProblemType == "Classification":
                            prediction = modelsDict[ip](batchDict[ip]['batch'].long().to(device)).argmax(axis=1)
                        
                        results = ( prediction == torch.tensor(batchDict[ip]['labels'], device=device) )
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
                if (total%1000000==0):                                               
                    ## Calculate remaining batches                        
                    for ip in batchDict:
                        batchCounter = batchDict[ip]['batchCounter']
                        batchDict[ip]['batchCounter']  = 0
                        prediction = 2.0                                                           
                        
                        if ProblemType == "Regression":
                            prediction = torch.sign(modelsDict[ip](batchDict[ip]['batch'].long().to(device)).cpu())
                            torch.nn.functional.relu(prediction, inplace=True)
                        elif ProblemType == "Classification":
                            prediction = (modelsDict[ip](batchDict[ip]['batch'].long().to(device)).argmax(axis=1).cpu())
                        
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
                batchCounter = batchDict[ip]['batchCounter']
                batchDict[ip]['batchCounter']  = 0
                prediction = 2.0                                                           
                if ProblemType == "Regression":
                    prediction = torch.sign(modelsDict[ip](batchDict[ip]['batch'].long().to(device)).cpu())
                    torch.nn.functional.relu(prediction, inplace=True)
                elif ProblemType == "Classification":
                    prediction = (modelsDict[ip](batchDict[ip]['batch'].long().to(device)).argmax(axis=1).cpu())                               
                
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
    # Close files and merge them together into one gz file.
    for ipKey in outputTraceDict:
        outputTraceDict[ipKey].close()
    merger(mode, trace)
    
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
        traceFolder = "path/to/Datasets/myTraces/{}".format(bench)
    elif(platform.system()=="Linux"):
        traceFolder = "/path/to/datasets/allBranches/{}".format(bench)
    
    tracesPaths = [
        "{}/{}".format(traceFolder, trace) for trace in os.listdir(traceFolder) if "ref" in trace
        ]
    # skip existing    
    existing = os.listdir(outputFolder)
    # CHECK IF EXISTS AND SKIP, DELETE LINE ABOVE    
    for trace in tracesPaths:
        simpoint = re.search("-([0-9]*?)B.", trace).group(1) # extract simpoint
        output = "{outputFolder}/{bench}-{simpoint}B.pt".format(
            outputFolder = outputFolder,
            bench = bench,
            simpoint = simpoint)          
        # SPECIAL: CHECK IF OUTPUT ALREADY EXISTS AND SKIP
        if os.path.basename(output) in existing: continue 
        # CHECK IF EXISTS AND SKIP, DELETE LINE ABOVE
        print("Now will run Benchmark: {}, Simpoint: {}B, with NeuralN: {}".format(
            bench, simpoint, NeuralNetwork
        ))              
        main(output, NeuralNetwork, bench, trace)    
    exit()
