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
#file_name = '/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'

n_out, n_hidden, learning_rate, n_iter, batch_size = 1, 8000, 0.01, 100, 800
n = 0
t = n_iter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self, n_input, n_hidden, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden).to(device)
        self.fc2 = nn.Linear(n_hidden, int(n_hidden/2)).to(device)
        self.fc3 = nn.Linear(int(n_hidden/2), int(n_hidden/2)).to(device)
        self.fc4 = nn.Linear(int(n_hidden/2), int(n_hidden/2)).to(device)
        #self.fc5 = nn.Linear(int(u/2), int(u/2)).to(device)
        self.fc5 = nn.Linear(int(n_hidden/2), 1).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        x = self.fc5(x)
        return x

for i in range(8,9):
    epoch_list = []
    input_d = []

    df = pd.read_csv(file_name.format(i))
    n_input = int(len(df.columns)-2)
    net = Net(n_input, n_hidden, n_out)
    
    net.to(device)

    target = df.iloc[:, 1]
    max = target.max()
    target = target.div(max)
    target = target.subtract(target.mean())

    amplitude = target.max() - target.min()
    target = target.div(amplitude)

    x_train = torch.FloatTensor(df.iloc[:, 2:len(df.columns)].values)
    x_train = x_train.add(1)
    x_train = np.log(x_train)
    
    target = target.values.reshape((-1,1))
    target = torch.FloatTensor(target)

    x_train = x_train.to(device)
    target = target.to(device)

    train_data = TensorDataset(x_train, target)

    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr = learning_rate)
    
    running_loss_list = []
    diff = []
    n = 0
    l = len(list(df.index))
    count = []

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