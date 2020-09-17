#randomly collect from big dataset

from random import randint
import torch


new = torch.zeros(5000, 201,2)

train = torch.load("./../Datasets/train_600_210B_600K.pt")

for i in range(len(new)):
    indx = randint(0,len(train)-1)
    new[i] = train[indx]

torch.save(new, "5000.pt")