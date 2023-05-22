import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

n_hidden, n_out, batch_size, learning_rate = 1000, 1, 10, 0.01

class Net(nn.Module):

    def __init__(self, n_input, n_hidden, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        #self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.fc3(x)
        return x
      
file_name = '/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
df = pd.read_csv(file_name.format(2),nrows=10)
df=df.dropna(how='all')
act_index=df.columns.get_loc('Act')
#df.Act.isna().sum()
#desc_dim=df.shape[1]-(act_index+1)
X=df.iloc[:,act_index+1:].values

X=np.log(X + 1)

# Normalize activations
Y = np.asarray(df.Act)
Y_mean = np.mean(Y)
Y_std = np.std(Y)

df.Act=(df.Act - Y_mean) / Y_std

# -> this is culprit Y=df.Act.values
Y=df.Act.values.reshape((-1,1))

x_train=torch.tensor(X).float()
y_train=torch.tensor(Y).float()


n_input=x_train.size()[-1]
model=Net(n_input, n_hidden, n_out) 
print(model)

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
for epoch in range(300):
    pred_y = model(x_train)
    loss = loss_function(pred_y, y_train)
    if epoch % 10 == 9 :
       print(loss, epoch)
       losses.append(loss.item())

    model.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
#plt.show(block=True) 
plt.savefig('loss.png')