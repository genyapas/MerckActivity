import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random2

#file_name = '/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name = '/root/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
file_name = '/home/ewgeni/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name = '/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'

n_out, n_hidden, learning_rate, n_iter, batch_size, n_input, dfa = 1, 8000, 0.01, 10, 8000, 11080, []
n = 0
t = n_iter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#fig, axes = plt.subplots(nrows = 2, ncols = 2)


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

def drop_prefix(self, prefix):
    self.columns = self.columns.str.lstrip(prefix)
    return self

epoch_lists = []
running_loss_lists = []

for i in range(1, 16):
    filename = file_name.format(i)
    n = sum(1 for line in open(filename)) - 1
    s = 1000
    skip = sorted(random2.sample(range(1,n+1),n-s))
    df = pd.read_csv(filename, skiprows=skip)
    dfa.append(df)

dfa = pd.concat(dfa, axis = 0)
dfa = dfa.fillna(0.0)

dfa1 = dfa.iloc[:, 0:2]

dfa2 = dfa.iloc[:, 2:len(dfa.columns)]
dfa2 = drop_prefix(dfa2, 'D_')
dfa2.columns = dfa2.columns.astype(int)
dfa2 = dfa2.reindex(sorted(dfa2.columns), axis = 1)

dfa = pd.concat([dfa1, dfa2], axis=1, join='inner')

epoch_list = []
input_d = []

net = Net(n_input, n_hidden, n_out)

net.to(device)

target = dfa.iloc[:, 1]
max = target.max()
target = target.div(max)
target_mean = target.mean()
target = target.subtract(target.mean())

amplitude = target.max() - target.min()
target = target.div(amplitude)

x_train = torch.FloatTensor(dfa.iloc[:, 2:len(dfa.columns)].values)
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
l = len(list(dfa.index))
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
epoch_lists.append(epoch_list)
running_loss_lists.append(running_loss_list)

##Feeing up memory
del x_train
del target

##Test

file_name = '/home/ewgeni/projects/MerckActivity/TestSet/ACT{}_competition_test.csv'
dfa = []
preds = []

for i in range(1, 16):
    filename = file_name.format(i)
    df = pd.read_csv(filename)
    dfa.append(df)

dfa = pd.concat(dfa, axis = 0)
dfa = dfa.fillna(0.0)

dfa1 = dfa.iloc[:, 0:1]

dfa2 = dfa.iloc[:, 1:len(dfa.columns)]
dfa2 = drop_prefix(dfa2, 'D_')
dfa2.columns = dfa2.columns.astype(int)
dfa2 = dfa2.reindex(sorted(dfa2.columns), axis=1)

dfa = pd.concat([dfa1, dfa2], axis=1, join='inner')

x_test = dfa.iloc[:, 1:len(dfa.columns)].values
x_test = x_test + 1
x_test = np.log(x_test)

x_test = torch.FloatTensor(x_test)

x_test = x_test.to(device)
counter = 0

preds = net(x_test).tolist()
preds = [(pred * amplitude + target_mean) * max for pred in preds]
print(preds, len(preds), type(preds))
