import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random2
import time
import os
os.environ['MPLCONFIGDIR'] = "/dev/shm/ipasichn"
import matplotlib.pyplot as plt

file_name = '/home/ipasichn/work/MerckActivity/TrainingSet/ACT{}_competition_training.csv'

n_out, n_hidden, learning_rate, n_iter, batch_size, n_input, dfa = 1, 8000, 0.01, 50, 8000, 11081, []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def drop_prefix(self, prefix):
    self.columns = self.columns.str.lstrip(prefix)
    return self

class Net(nn.Module):

    def __init__(self, n_input, n_hidden, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden).to(device)
        self.fc2 = nn.Linear(n_hidden, int(n_hidden/2)).to(device)
        self.fc3 = nn.Linear(int(n_hidden/2), int(n_hidden/2)).to(device)
        self.fc4 = nn.Linear(int(n_hidden/2), int(n_hidden/2)).to(device)
        self.fc5 = nn.Linear(int(n_hidden/2), 1).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

net = Net(n_input, n_hidden, n_out)
net= nn.DataParallel(net)
net.load_state_dict(torch.load('/home/ipasichn/work/MerckActivity/MerckActivity100723.pt'))
net.to(device)

dfa = []
preds = []

for i in range(1, 16):
    filename = file_name.format(i)
    df = pd.read_csv(filename)
    target_molecule = [i] * len(df)
    df['target_molecule'] = target_molecule
    dfa.append(df)

dfa = pd.concat(dfa, axis = 0)
dfa = dfa.fillna(0.0)

target_molecule = dfa.pop('target_molecule')
activities = dfa.pop('Act')
dfa.insert(1, target_molecule.name, target_molecule)

dfa1 = dfa.iloc[:, 0:2]

dfa2 = dfa.iloc[:, 2:len(dfa.columns)]
dfa2 = drop_prefix(dfa2, 'D_')
dfa2.columns = dfa2.columns.astype(int)
dfa2 = dfa2.reindex(sorted(dfa2.columns), axis=1)

dfa = pd.concat([dfa1, dfa2], axis=1, join='inner')

x_test = dfa.iloc[:, 1:len(dfa.columns)].values
x_test = x_test + 1
x_test = np.log(x_test)

x_test = torch.FloatTensor(x_test)

x_test = x_test.to(device)

preds = net(x_test).tolist()
activities = activities.values.tolist()

preds = [(pred * amplitude + target_mean) * max for sublist in preds for pred in sublist]

dif = []
for i in range(len(preds)):
    dif.append(preds[i] - activities[i]) 

print('\n' + 'Trainingtime: ' + str(tt) + 's', 'for ' + str(n_iter - 1) + ' epochs', abs(sum(dif)))