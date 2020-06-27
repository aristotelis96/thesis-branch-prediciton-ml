import numpy as np
import gzip
import torch
from torch.utils.data import Dataset


## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")

class BranchDataset(Dataset):
    def __init__(self, npy, transform=None):
        #self.data = np.load(npyFile)
        self.data = npy

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
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
            label = torch.tensor(0, dtype=torch.float64).clone().detach()
        else:
            label = torch.tensor(1, dtype=torch.float64).clone().detach()
        return sample, label

def read(file, start=0, end=1000000):
    #print(start,end)
    data = torch.zeros([end-start+1,201,2], dtype=torch.float64)

    with gzip.open(file, 'rt') as fp:  
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
                data[sample-start, history,0] = np.float64(ip)
                data[sample-start,history,1] = np.float64(taken)
                history += 1           
        except Exception as e:
            print("ERROR" ,e, general)
            
            
    print(sample)                
    return data
