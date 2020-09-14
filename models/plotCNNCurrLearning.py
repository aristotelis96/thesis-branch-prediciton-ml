import matplotlib.pyplot as plt
import torch

lossT, lossV, accT, accV = [],[],[],[]

ck = torch.load('checkpointCNN.pt')


plt.plot(ck['total_Train_loss'], label='LossTrain' ,color='gray')
plt.plot(ck['total_Train_accuracy'], label='AccTrain' ,color='lime')
plt.plot(ck['total_Validation_loss'], label='LossValid' ,color='black')
plt.plot(ck['total_Validation_accuracy'], label='AccValid' ,color='red')
plt.legend(loc='right')
plt.savefig("currentLearning.png")
