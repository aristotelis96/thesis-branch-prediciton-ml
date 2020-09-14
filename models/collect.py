import torch

new = torch.zeros(200000,201,2)

train = torch.load("../Datasets/train_600_210B_600K.pt")

a = {}

for i in range(len(train)):
    if int(train[i][0][0]) not in a:
        a[int(train[i][0][0])] = {0: 0, 1:0}
    else:
        if train[i][0][1] ==0:
            a[int(train[i][0][0])][0] += 1
        else:
            a[int(train[i][0][0])][1]+=1

j=0
for key in a:
    if(key!=0):
        sum0=0
        sum1=0
        for i in range(len(train)):
            if int(train[i][0][0])==key:
                if(sum0<a[key][0] and sum0<(len(new)/24) and train[i][0][1]==0):
                    new[j] = train[i]
                    j+=1
                    sum0+=1
                if(sum1<a[key][1] and sum1<(len(new)/24) and train[i][0][1]==1):
                    new[j] = train[i]
                    j+=1
                    sum1+=1

torch.save(new, "uniform.pt")

