import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

file_name='/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name='/Users/genya/projects/MerckActivity/TestSet/ACT{}_competition_test.csv'
#file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
u=10
n=0
epoch_list=[]
loss_list=[]
batch_size=100

class Net(nn.Module):

    def __init__(self, y):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(y, int(y/2))
        #self.fc1 = nn.Linear(y, 15)
        self.fc2 = nn.Linear(int(y/2), 1)
        #self.fc2 = nn.Linear(15, 1)
        #self.fc3 = nn.Linear(u, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

for i in range(7,8):
    input_d=[]
    df=pd.read_csv(file_name.format(i))
    input_d=int(len(df.columns)-2)
    #input_d=100
    #y_train = (torch.rand(size=(batch_size, 1)) < 0.5).float()
    #x_train = torch.randn(batch_size, input_d)
    #print(df.head)
    net=Net(y=input_d)
    #print(net)
    x_train=torch.FloatTensor(df.iloc[:,2:len(df.columns)].values)
    #x_train=torch.FloatTensor(df.iloc[:,2:len(df.columns)].sample(n=100, axis=1).values)
    #print(x_train)
    #print(input, len(df.columns))
    target=torch.FloatTensor(df.iloc[:,1].values)
    print(target)
    #target=torch.FloatTensor(y_train)
    #target=target.view(len(df.index), 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(50):
        #running_loss = 0.0
        output=net(x_train)
        loss = criterion(output, target)
        if epoch % 10==9:
            print(loss.item(), epoch, output)
            #layer = net.fc2.state_dict()
            #print(layer['weight'])
            #print(layer['bias'])
            loss_list.append(loss.item())
            epoch_list.append(epoch)
        #net.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plt.scatter(epoch_list, loss_list)
    plt.show()
    #x->epoch, y->loss
        #running_loss += loss.item()
        #if n % 10 == 9:
            #print(f'[{epoch + 1}, {n + 1:5d}] loss: {running_loss / 10:.3f}')
            #running_loss = 0.0
    #print('Finished Training')