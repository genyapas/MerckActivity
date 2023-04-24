import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

file_name='/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name='/Users/genya/projects/MerckActivity/TestSet/ACT{}_competition_test.csv'
#file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
u=1000
n=0
epoch_list=[]
loss_list=[]

class Net(nn.Module):

    def __init__(self, y):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(y, int(y/2))
        self.fc2 = nn.Linear(int(y/2), u)
        self.fc3 = nn.Linear(u, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

for i in range(7,8):
    input_d=[]
    df=pd.read_csv(file_name.format(i))
    input_d=int(len(df.columns)-2)
    net=Net(y=input_d)
    #print(net)
    input=torch.FloatTensor(df.iloc[:,2:len(df.columns)].values)
    #print(input, len(df.columns))
    target=torch.FloatTensor(df.iloc[:,1].values)
    target=target.view(len(df.index), 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(100):
        #running_loss = 0.0
        output=net(input)
        loss = criterion(output, target)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        epoch_list.append(epoch)
    plt.scatter(epoch_list, loss_list)
    plt.show()
    #x->epoch, y->loss
        #running_loss += loss.item()
        #if n % 10 == 9:
            #print(f'[{epoch + 1}, {n + 1:5d}] loss: {running_loss / 10:.3f}')
            #running_loss = 0.0
    #print('Finished Training')