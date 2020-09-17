import torch

size = 5000

new = torch.zeros(size,201,2)

train = torch.load("631.pt")

a = {}

for i in range(len(train)):
    if int(train[i][0][0]) not in a:
        a[int(train[i][0][0])] = {0: 0, 1:0}
    else:
        if train[i][0][1] ==0:
            a[int(train[i][0][0])][0] += 1
        else:
            a[int(train[i][0][0])][1]+=1
distinctH2Ps = len(a)
j=0
for key in a:    
    if(key!=0):
        sum0=0
        sum1=0
        for i in range(len(train)):
            if int(train[i][0][0])==key:
                if(sum0<a[key][0] and sum0<(len(new)/(distinctH2Ps*2)) and train[i][0][1]==0):
                    new[j] = train[i]
                    j+=1
                    sum0+=1
                if(sum1<a[key][1] and sum1<(len(new)/(distinctH2Ps*2)) and train[i][0][1]==1):
                    new[j] = train[i]
                    j+=1
                    sum1+=1
# narrow 0 sammples not used                    
if(j<size):
    new = torch.narrow(new, 0, 0, j)
torch.save(new, "631uni.pt")

