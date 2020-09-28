#randomly collect from big dataset

from random import randint
import torch
import gzip
import numpy as np


train = torch.load("./../Datasets/train_600_210B_600K.pt")
valid = torch.load("./../Datasets/valid_600_210B_600K-800K.pt")
train = torch.cat((train,valid), dim=0)

HP2s = [4224005,4957196,4976121,5004294]
H2Pbetter = [4978645,5008304,5307410,4969208,5015393]
allH2Ps = [5015393, 4224005, 5004294, 4999560, 4295273, 4957196, 5008304, 5307410, 4978645, 4969208, 4976121, 4226075]

using = allH2Ps

#new = torch.zeros(5000*len(using), 201,2)
new = torch.zeros(5000, 201,2)
# with gzip.open("../Datasets/631.deepsjeng_s-928B.champsimtrace.xz._.dataset_unique.txt.gz", 'rt') as fp:
#     line=""
#     for line in fp:
#         if "--- H2P ---" not in line:
#             continue
#         break
#     history = 0
#     sample = 0
#     step=0
#     skip=4200
#     for line in fp:
#         if line.startswith("Finished"):
#             break
#         if "--- H2P ---" in line or "\n"==line or line.startswith("Warmup complete"):            
#             step+=1
#             if(step==skip):
#                 sample+=1
#                 history=0            
#             if(step>skip):
#                 step=0                            
#             continue 
#         if(step<skip):
#             continue  
#         [ip, taken] = line.split(" ")
#         new[sample, history,0] = np.float64(ip)
#         new[sample, history, 1] = np.float64(taken)
#         history+=1

# for cnt,H2P in enumerate(using):
#     for i in range(5000):
#         while True:
#             indx = randint(0,len(train)-1)
#             if(int(train[indx][0][0]) in using):
#                 break
#         new[i+5000*cnt] = train[indx]
for i in range(5000):
    while True:
        indx = randint(0,len(train)-1)
        if(int(train[indx   ][0][0]) in using):
            break
    new[i] = train[indx]

newValid = torch.zeros(40000,201,2)
for i in range(len(newValid)):
    while True:
        indx = randint(0, len(valid)-1)
        if(valid[indx][0][0] in using):
            break
    newValid[i]=valid[indx]

torch.save(new, "600_210B_ALL5000.pt")
torch.save(newValid, "NewValidation.pt")