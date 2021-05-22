import numpy as np
import gzip
import torch
from torch.utils.data import Dataset
import os
import time
from collections import deque

## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")

# Dataset Folder #
DatasetDir="C:/Users/Aristotelis/Desktop/diploma/Datasets/myTraces/531.deepsjeng/history582/1000/"

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

    

class BranchDataset(Dataset):
    def __init__(self, npy, transform=None, encodePCList=False, historyLen=200, pc_bits=7, concatenate_dir=True):
        #self.data = np.load(npyFile)
        self.data = npy
        self.encodePCList=encodePCList
        self.pc_bits = pc_bits
        self.concatenate_dir = concatenate_dir
        self.historyLen = historyLen
        self.encoder = Encoder(self.pc_bits, self.concatenate_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):    
        if(self.encodePCList):
            sample = self.data[idx][1:].clone().detach()
            encodedPCArray = [0]*self.historyLen
            #encodedPCArray = torch.zeros((582), dtype=torch.int64)
            for i,s in enumerate(sample):
                pc = int(s[0])                
                label =  int(self.data[idx][i+1][1])
                encodedPC = self.encoder.encode(pc, label)
                encodedPCArray[i] = encodedPC
                #encodedPCArray[i][1] = s[1] #taken/nottaken
                #encodedPCArray[i][1] = self.data[idx][i+1][1]
                #s = encodedPCArray[i].clone().detach()
                #s[0] = encodedPC
                sample = torch.tensor(encodedPCArray)


        else:
            sample = self.data[idx][1:].clone().detach() 
            for s in sample:
                if(s[1]==0.0):
                    s[0], s[1] = 1.0, 0.0
                else:
                    s[0], s[1] = 0.0 , 1.0
        #sample[0][1]=0.5
        #sample = sample.flip([0])  
        label = self.data[idx][0][1].clone().detach()
        if(label==0):
            label = torch.tensor(-1, dtype=torch.float64).clone().detach()
        else:
            label = torch.tensor(1, dtype=torch.float64).clone().detach()
        return sample, label

def read(file, start=0, end=100000):
    #print(start,end)
    data = torch.zeros([end-start+1,583,2], dtype=torch.float64)
    #data = [[[0]*2]*583]*(end-start)
    with gzip.open(DatasetDir + file, 'rt') as fp:  
        general =0
        try:
            for line in fp:
                if "--- H2P ---" not in line:
                    general += 1
                    continue
                break
            history=0
            sample = 0
            for line in fp:    
                general+=1
                if line.startswith("Finished"):
                    break          
                if "--- H2P ---" in line or "\n"==line or line.startswith("Warmup complete"):
                    sample+=1
                    history=0
                    continue
                if(sample<start):
                    continue
                if(sample>=end):
                    break
                [ip, taken] = line.split(" ")
                data[sample-start-1, history,0] = np.float64(ip)
                data[sample-start-1,history,1] = np.float64(taken)
                #data[sample-start-1][history][0] = np.float64(ip)
                #data[sample-start-1][history][1] = np.float64(taken)
                history += 1           
        except Exception as e:
            print("ERROR" ,e, general)
            
            
    #print(sample)                
    return torch.tensor(data, dtype=torch.float64)

def readTrainValid(file, start=0, end=100000, ratio=0.6):
    middle = int((end-start)*ratio)
    train = read(file, start, start+middle)
    valid = read(file, start+middle+1, end)
    return train, valid

def DatasetLength(file):
    with gzip.open(DatasetDir + file, 'rt') as fp:  
        general =0        
        try:
            for line in fp:
                if "--- H2P ---" not in line:
                    general += 1
                    continue
                break
            history=0
            sample = 0
            for line in fp:    
                general+=1
                if line.startswith("Finished"):
                    break          
                if "--- H2P ---" in line or "\n"==line or line.startswith("Warmup complete"):
                    sample+=1
                    history=0
                    continue            
        except Exception as e:
            print("ERROR" ,e, general)    
    return sample


def readFileList(fileList, start=0, end=50000, ratio=1.0):
    train = torch.zeros((1,583,2), dtype=torch.float64)
    valid = torch.zeros((1,583,2), dtype=torch.float64)        
    for file in fileList:
        print(file)    
        end = DatasetLength(file)-1
        print(end)
        tempTrain, tempValid = readTrainValid(file, start, end, ratio)
        train = torch.cat((train, tempTrain))
        valid = torch.cat((valid, tempValid))
    return train, valid

def readFileListWriteToDisk(fileList, start=0, end=50000, ratio=1.0):
    train = torch.zeros((1,583,2), dtype=torch.float64)
    valid = torch.zeros((1,583,2), dtype=torch.float64)        
    for file in fileList:
        s=time.time()
        print(file)    
        end = DatasetLength(file)-1
        print(end)
        data, _ = readTrainValid(file, start, end, ratio)
        print(data.shape)
        path = "C:/Users/Aristotelis/Desktop/diploma/models/specificBranch/531.deepsjeng/datasetsBigHistory/"
        branches = set()        
        for sample in data:
            branches.add(int(sample[0][0])) 
        branches.discard(0)
        limit = 800
        for branch in branches:
            print(branch)
            length = len([x for x in data if int(x[0][0])==int(branch)])
            print(length)
            # if(length< limit): 
            #     print('discarding branch, not enough information')
            #     continue
            new = torch.zeros(length, 583, 2)            
            j=0
            for i in range(len(data)):
                if int(data[i][0][0])==int(branch):
                    new[j]=data[i]
                    j+=1
            
            print(len(new))
            fileName = path+str(branch)+".pt"
            if(os.path.exists(fileName)):
                oldfile = torch.load(fileName)
                newFile = torch.cat((oldfile, new))
                torch.save(newFile, fileName)    
            else:
                torch.save(new, fileName)
        print("Time for ", fileName, time.time()-s)
    return 
