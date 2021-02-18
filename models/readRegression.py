import numpy as np
import gzip
import torch
from torch.utils.data import Dataset
import os

## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")

# Dataset Folder #
DatasetDir="C:/Users/Aristotelis/Desktop/diploma/Datasets/myTraces/531.deepsjeng/"
class BranchDataset(Dataset):
    def __init__(self, npy, transform=None, encodePCList=False):
        #self.data = np.load(npyFile)
        self.data = npy
        self.encodePCList=encodePCList

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):    
        if(self.encodePCList):
            sample = self.data[idx][1:].clone().detach()
            encodedPCArray = torch.zeros((200), dtype=torch.int64)
            for i,s in enumerate(sample):
                pc = int(s[0])
                # Take 7 LSB and shift 1 possition
                encodedPC = (pc & 0b1111111) << 1
                # Add taken or not taken
                encodedPC += int(self.data[idx][i+1][1])
                encodedPCArray[i] = encodedPC
                #encodedPCArray[i][1] = s[1] #taken/nottaken
                #encodedPCArray[i][1] = self.data[idx][i+1][1]
                #s = encodedPCArray[i].clone().detach()
                #s[0] = encodedPC
                sample = encodedPCArray.clone().detach()


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
            label = torch.tensor([0], dtype=torch.float64).clone().detach()
        else:
            label = torch.tensor([1], dtype=torch.float64).clone().detach()
        return sample, label

def read(file, start=0, end=100000):
    #print(start,end)
    data = torch.zeros([end-start+1,201,2], dtype=torch.float64)

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
                history += 1           
        except Exception as e:
            print("ERROR" ,e, general)
            
            
    #print(sample)                
    return data

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


def readFileList(fileList, start=0, end=50000, ratio=0.0):
    train = torch.zeros((1,201,2), dtype=torch.float64)
    valid = torch.zeros((1,201,2), dtype=torch.float64)        
    for file in fileList:
        print(file)    
        end = DatasetLength(file)-1
        print(end)
        tempTrain, tempValid = readTrainValid(file, start, end, ratio)
        train = torch.cat((train, tempTrain))
        valid = torch.cat((valid, tempValid))
    return train, valid
