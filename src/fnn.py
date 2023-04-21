import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

file_name='/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name='/Users/genya/projects/MerckActivity/TestSet/ACT{}_competition_test.csv'
#file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
input_d=[]
u=5
#n=11

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
    
    #df=pd.read_csv(file_name.format(i), nrows=1, usecols=[i for i in range(1,n)])
    cols=list(pd.read_csv(file_name.format(i), nrows=1))
    df=pd.read_csv(file_name.format(i), usecols=lambda x: x not in ['MOLECULE', 'Act'], nrows=1)
    for col in df.columns:
        input_d.append(col)
    net=Net(y=int(len(input_d)))
    print(net)    
    
    df=pd.read_csv(file_name.format(i), usecols=lambda x: x not in ['MOLECULE', 'Act'])
    #df=pd.read_csv(file_name.format(i), usecols=[i for i in range(1,n)])
    data=torch.FloatTensor(df.values)
    print(data)