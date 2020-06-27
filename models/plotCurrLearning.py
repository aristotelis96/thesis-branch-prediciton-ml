import matplotlib.pyplot as plt

lossT, lossV, accT, accV = [],[],[],[]

with open("currLearning.txt") as fp:
    for line in fp:
        if "train" in line:
            words = line.split(" ")
            lossT.append(float(words[5]))
            accT.append(float(words[-1]))
        if "valid" in line:
            words = line.split(" ")
            lossV.append(float(words[5]))
            accV.append(float(words[-1]))



plt.plot(lossT, label='lossTrain' ,color='red')
plt.plot(accT, label='accTrain' ,color='black')
plt.plot(lossV, label='lossValid' ,color='orange')
plt.plot(accV, label='accValid' ,color='green')
plt.legend()
plt.savefig("currentLearning.png")
