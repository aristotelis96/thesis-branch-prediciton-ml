#randomly collect from big dataset

from random import randint
import torch
import gzip
import numpy as np

new = torch.zeros(5000, 201,2)

#train = torch.load("./../Datasets/train_600_210B_600K.pt")
#valid = torch.load("./../Datasets/valid_600_210B_600K-800K.pt")
#train = torch.cat((train,valid), dim=0)

with gzip.open("../Datasets/631.deepsjeng_s-928B.champsimtrace.xz._.dataset_unique.txt.gz", 'rt') as fp:
    line=""
    for line in fp:
        if "--- H2P ---" not in line:
            continue
        break
    history = 0
    sample = 0
    step=0
    skip=4200
    for line in fp:
        if line.startswith("Finished"):
            break
        if "--- H2P ---" in line or "\n"==line or line.startswith("Warmup complete"):            
            step+=1
            if(step==skip):
                sample+=1
                history=0            
            if(step>skip):
                step=0                            
            continue 
        if(step<skip):
            continue  
        [ip, taken] = line.split(" ")
        new[sample, history,0] = np.float64(ip)
        new[sample, history, 1] = np.float64(taken)
        history+=1

# for i in range(len(new)):
#     indx = randint(0,len(train)-1)
#     new[i] = train[indx]

torch.save(new, "631_5000.pt")