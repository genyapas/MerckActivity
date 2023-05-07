import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

file_name='/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name='/Users/genya/projects/MerckActivity/TestSet/ACT{}_competition_test.csv'
#file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
u=10
n=0
epoch_list=[]
n_iter=500

class Net(nn.Module):

    def __init__(self, y):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(y, int(y/2))
        self.fc1 = nn.Linear(y, 100)
        self.fc2 = nn.Linear(100, 1)
        #self.fc3 = nn.Linear(int(y/3), 1)
        #self.fc3 = nn.Linear(int(y/2), 1)
        #self.fc4 = nn.Linear(400, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

for i in range(7,8):
    input_d = []
    df = pd.read_csv(file_name.format(i))
    input_d = int(len(df.columns)-2)
    net = Net(y = input_d)

    target = df.iloc[:, 1:len(df.columns)]
    max = df.max()
    target.div(max)

    x_train = torch.FloatTensor(df.iloc[:, 2:len(df.columns)].values)
    target = torch.FloatTensor(target.values)
    train_data = TensorDataset(x_train, target)

    # create data loader
    train_loader = DataLoader(dataset = train_data, batch_size = 400)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.01)
    running_loss_list = []

    for epoch in range(n_iter):
        running_loss = 0

        for x_train, target in train_loader:
            optimizer.zero_grad()
            output = net(x_train)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 100 == 99:
            print(running_loss, epoch)
            epoch_list.append(epoch)
            running_loss_list.append(running_loss)

    plt.scatter(epoch_list, running_loss_list)
    plt.show()