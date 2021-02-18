import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.utils.data import DataLoader
import time
import read
from read import BranchDataset
import os

def init_lstm(lstm, lstm_hidden_size, forget_bias=2):
    for name,weights in lstm.named_parameters():
        if "bias_hh" in name:
            #weights are initialized 
            #(b_hi|b_hf|b_hg|b_ho), 
            weights[lstm_hidden_size:lstm_hidden_size*2].data.fill_(forget_bias)
        elif 'bias_ih' in name:
            #(b_ii|b_if|b_ig|b_io)
            pass
        elif "weight_hh" in name:
            torch.nn.init.orthogonal_(weights)
        elif 'weight_ih' in name:
            torch.nn.init.xavier_normal_(weights)
    return lstm


class RNNLayer(nn.Module):
    def __init__(self,  input_dim, out_dim, num_layers, init_weights = True, batch_first=True, rnn_layer = 'gru', normalization = False):
        super().__init__()        

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.rnn_layer = rnn_layer
        
        self.E = torch.eye(self.input_dim, dtype=torch.float32)        

        self.embedding1 = nn.Embedding(128,64)
        self.embedding2 = nn.Embedding(2,64)

        if rnn_layer == 'lstm':
            self.rnn = nn.LSTM(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
            self.fc = nn.Linear(out_dim*200, 2)
        elif rnn_layer == 'gru':
            self.rnn = nn.GRU(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
        elif rnn_layer =='transformer':
            #self.inpLinear = nn.Linear(2, self.input_dim)
            self.rnn = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.out_dim)
            self.fc = nn.Linear(input_dim*200, 2) # nn.Linear(out_dim*200, 2)
        else:
            raise NotImplementedError(rnn_layer)
        
        if init_weights:
            self.rnn = init_lstm(self.rnn, self.out_dim)

        if (normalization):
            self.normalization = nn.BatchNorm1d(out_dim)
        else:
            self.normalization = None
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, seq):
        bsize = seq.size(0)        
        
        #data = self.E[seq.data.type(torch.long)]
        #seq = data.to(torch.device("cuda:0"))
        lay1=self.embedding1(seq.data[:,:,0].type(torch.long))
        lay2=self.embedding2(seq.data[:,:,1].type(torch.long))
        seq = torch.cat((lay1,lay2),2)

        if self.rnn_layer == 'transformer':
            #seq = self.inpLinear(seq)
            self.rnn_out = self.rnn(seq)                   
            #self.rnn_out = self.rnn_out[:,-1,:]  
            self.rnn_out = self.rnn_out.reshape(len(self.rnn_out), 200*self.input_dim)            
        elif self.rnn_layer =='lstm':
            self.rnn_out, (self.h, self.c) = self.rnn(seq)                      
            self.rnn_out = self.rnn_out.flatten(1)                     
        elif self.rnn_layer == 'gru': 
            self.rnn_out, self.h = self.rnn(seq)

        if self.normalization:
            self.rnn_out = self.normalization(self.rnn_out)
        
        out = self.fc(self.rnn_out)   

        ## out = self.fc(self.rnn_out.view(bsize,-1))        
        return self.softmax(out)

## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")

def main(branch):
    ## Parameters ##
    n_class = 128
    num_layers = 1
    normalization = False
    input_dim = 128
    rnn_layer = 'transformer'

    # learning Rate
    learning_rate = 1e-4

    # Load a checkpoint?
    ContinueFromCheckpoint = False

    # Epoch
    epochStart = 0
    epochEnd = 40

    ## Model 
    model = RNNLayer(rnn_layer=rnn_layer, input_dim=input_dim, out_dim=n_class, num_layers=num_layers, normalization=normalization).to(device)

    print(model)
    ## TRAIN/TEST DATALOADER
    # Parameters
    paramsTrain = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 2}
    paramsValid = {'batch_size': 64,
            'shuffle': False,
            'num_workers': 2}

    # Benchmark
    input_bench = ["600.perlbench_s-210B.champsimtrace.xz._.dataset_unique.txt.gz"]
    startSample, endSample = 100, 400
    ratio = 0.75
    encodePCList=True
    loadPt=True
    inputDescription = 'sequence of 200 tuples (encodedPC *8bits*, Taken/NotTaken)'


    ##
    ## Optimize 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate/10)

    total_Train_loss = []
    total_Train_accuracy = []
    total_Validation_loss = []
    total_Validation_accuracy = []


    if ContinueFromCheckpoint:
        try:
            checkpoint = torch.load("checkpoint.pt")
            epochStart = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            total_Train_loss = checkpoint['total_Train_loss']
            total_Train_accuracy = checkpoint['total_Train_accuracy']
            total_Validation_loss = checkpoint['total_Validation_loss']
            total_Validation_accuracy = checkpoint['total_Validation_accuracy']
        except:
            print("FAILED TO LOAD FROM CHECKPOINT, CHECK FILE")
            exit()

    print("Loading TrainDataset")
    print("Loading ValidationDataset")

    if(loadPt):
        #train, valid = torch.load("./5000.pt"), torch.load("./../Datasets/valid_600_210B_600K-800K.pt")
        train = torch.load("./specificBranch/datasets/250/"+str(branch)+".pt")
        valid = torch.chunk(train, 200, dim=0)[0]
        #train, valid = torch.split(train, [len(train)-10000,10000])
        #valid = torch.load("./NewValidation.pt")  
        #valid = torch.load("NewValidation.pt")
    else:
        train, valid = read.readFileList(input_bench, startSample,endSample, ratio=ratio)
        #torch.save(train, "train_600_210B_600K.pt")
        #torch.save(valid, "valid_600_210B_600K-800K.pt")


    training_set, validation_set = BranchDataset(train, encodePCList=encodePCList), BranchDataset(valid, encodePCList=encodePCList)

    train_loader = DataLoader(training_set, **paramsTrain)
    valid_loader = DataLoader(validation_set, **paramsValid)
    s=0
    for i in range(len(train)):
        if(train[i][0][1]==0):
            s+=1
    print("0", s/len(train), "1", 1-s/len(train))
    #TRAINING
    now = time.time()
    epoch = epochStart
    while epoch < epochEnd:
        try:
            print("-------")
            #print("Epoch : " + str(epoch))
            loss_values = []
            running_loss = 0.0
            correct = 0.0
            for i, (X, labels) in enumerate(train_loader):
                model.train() 
                
                X = X.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(X.float())

                loss = criterion(outputs, labels.long()) 

                

                loss.backward()

                optimizer.step()
                # print statistics
                running_loss += loss.item()
                correct += float((outputs.argmax(axis=1).cpu() == labels.cpu()).sum())
            train_acc = correct/float(len(training_set))
            train_loss = running_loss/float(len(train_loader))
            total_Train_accuracy.append(train_acc)
            total_Train_loss.append(train_loss)
            print("Epoch: ", epoch, "train loss:", train_loss, " acc:", train_acc)
            #if(correct/float(len(training_set))>0.99):
                #break
            correct = 0
            for X_val, Validlabels in valid_loader:
                model.eval() 

                X_val = X_val.to(device)
                Validlabels = Validlabels.to(device)

                outputs = model(X_val.float())

                loss = criterion(outputs, Validlabels.long())

                loss_values.append(loss.item())    
            
                correct += (outputs.argmax(axis=1).cpu() == Validlabels.cpu()).sum()
            epochLoss = float(sum(loss_values))/float(len(valid_loader))
            total_Validation_loss.append(epochLoss)

            epochAccuracy = float(correct)/float(len(validation_set))
            total_Validation_accuracy.append(epochAccuracy)
            print("Epoch: ", epoch, "validation: loss:", epochLoss ,"acc:" , epochAccuracy)
#            if(epochAccuracy>0.99):
#                break
                # ADD METRIC (ACCURACY? Note batching)
            ## SAVE A CHECKPOINT as "checkpoint.pt"
            torch.save({
                    'epoch': epoch,
                    'learning_rate': learning_rate,
                    'model_architecture': str(model),
                    'model_specifics': {
                        'n_class' : n_class,
                        'num_layers' : num_layers,
                        'normalization' : normalization,
                        'input_dim' : input_dim,
                        'rnn_layer' : rnn_layer
                    },
                    'input_Data': {
                        'input_bench': input_bench,
                        'samples': {
                            'startSample': startSample,
                            'endSample': endSample,
                            'ratio': ratio                       
                        },
                        'encodePCList': encodePCList,
                        'type': inputDescription,
                        'batch_size': paramsTrain['batch_size']
                    },
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_Validation_accuracy': total_Validation_accuracy,
                    'total_Validation_loss': total_Validation_loss,
                    'total_Train_loss': total_Train_loss,
                    'total_Train_accuracy': total_Train_accuracy
                }, "./specificBranch/models/Transformer/250/Transformer"+str(branch)+".pt")
        except Exception as e:
            print("Error occured, reloading model from previous iteration and skipping epoch increase")
            print(e)
            checkpoint = torch.load("checkpoint.pt")
            epochStart = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            total_Train_loss = checkpoint['total_Train_loss']
            total_Train_accuracy = checkpoint['total_Train_accuracy']
            total_Validation_loss = checkpoint['total_Validation_loss']
            total_Validation_accuracy = checkpoint['total_Validation_accuracy']        
            continue
        epoch+=1 
        
    end = time.time()
    print("total time taken to train: ", end-now)

    print("Finish")


if __name__ == "__main__":
    branches = os.listdir("./specificBranch/datasets/250")
    branches.remove("all.pt")
    for branch in branches:
        print(branch)
        main(branch[:-3])
    #main(sys.argv[1])