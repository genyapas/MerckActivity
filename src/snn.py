import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

#file_name = '/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name = '/root/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
file_name = '/home/ewgeni/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name = '/Users/genya/projects/MerckActivity/TestSet/ACT{}_competition_test.csv'
#file_name = '/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'

u = 5000
n = 0
n_iter = 500

class Net(nn.Module):

    def __init__(self, y):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(y, int(y/2))
        self.fc1 = nn.Linear(y, u)
        self.fc2 = nn.Linear(u, int(u/2))
        self.fc3 = nn.Linear(int(u/2), int(u/2))
        #self.fc3 = nn.Linear(int(y/2), 1)
        self.fc4 = nn.Linear(int(u/2), 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

for i in range(8,9):
    epoch_list = []
    input_d = []

    df = pd.read_csv(file_name.format(i))
    input_d = int(len(df.columns)-2)
    net = Net(y = input_d)

    target = df.iloc[:, 1]
    max = target.max()
    target = target.div(max)
    target = target.subtract(target.mean())

    amplitude = target.max() - target.min()
    target = target.div(amplitude)

    x_train = torch.FloatTensor(df.iloc[:, 2:len(df.columns)].values)
    x_train = x_train.add(1)
    x_train = np.log(x_train)
    
    target = torch.FloatTensor(target.values)
    train_data = TensorDataset(x_train, target)

    train_loader = DataLoader(dataset = train_data, batch_size = 800, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.03, momentum=0.9, weight_decay=0.001)
    #optimizer = optim.SGD(net.parameters(), lr = 0.03, momentum=0.9, weight_decay=0.001)
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

        if epoch % 10 == 9:
            print(running_loss, epoch)
            epoch_list.append(epoch)
            running_loss_list.append(running_loss)

    plt.scatter(epoch_list, running_loss_list)
    plt.show()