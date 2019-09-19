# import libraries ---------------------------------------------
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

from sklearn import metrics
import matplotlib.pyplot as plt
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# read csv file ---------------------------------------------
df = pd.read_csv('/sample_data/odd_nums_labelled.csv')
df.head()

# configuring hypermeters -----------------------------------
batch_size = 20
num_epochs = 3
learning_rate = 0.1
weight_decay = 1e-4

# preprocess data and feed them into data loadesr -----------
x = np.array(df['data'].values, dtype=np.uint8).reshape((len(df), 1))
x = np.unpackbits(x, axis=1) # change decimal to binary
y = np.array(df['label'].values)
print(x.shape)
print(y.shape)

# create data_loaders
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

dataset = TensorDataset(x_tensor, y_tensor)

train_size = int(len(dataset)/5*4) # 20% used for validation
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset))

# model -----------------------------------------------------
# model with one hidden layer and one output layer
class my_model(nn.Module):
  def __init__(self, n_in=8, n_hidden=10, n_out=2):
    super(my_model,self).__init__()
    self.n_in  = n_in
    self.n_out = n_out

    self.layer1 = nn.Linear(self.n_in, self.n_out, bias=True)  # hidden layer
    self.logprob = nn.LogSoftmax(dim=1)                        # -log(softmax probability)

  def forward(self, x):
    x = self.layer1(x)
    x = self.logprob(x)
    return x
  
  def print_weights(self):
    for param in self.parameters():
      print(param.data)

# model
model = my_model()

# loss function
criterium = nn.NLLLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# training model ------------------------------------------------
# for checking how the weights changes
iteration_array = []
loss_array = []
weights_array = []

for i in range(num_epochs):
  # training for one epoch
  for k, (data, target) in enumerate(train_loader):
    # definition of inputs as variables for the net
    # requires_grad is set False because we do not need to compute the derivative of the inputs
    data   = Variable(data, requires_grad=False)
    target = Variable(target.long(), requires_grad=False)

    # set gradient to 0
    optimizer.zero_grad()
    # feed forward
    pred = model(data)
    # loss calculation
    loss = criterium(pred, target.view(-1))
    # gradient calculation.
    loss.backward()

    iteration_array.append(batch_size*i+k)
    loss_array.append(loss.item())
    weights_array.append([param.data.tolist() for param in model.parameters()])
    # print loss every iterations
    if k%1==0:
      print('Loss {:.4f} at iter {:d}'.format(loss.item(), batch_size*i+k))

    # model weight modification based on the optimizer. 
    optimizer.step()

  # model.print_weights()

# testing -------------------------------------------------
for k, (data, target) in enumerate(val_loader):
  # definition of inputs as variables for the net
  # requires_grad is set False because we do not need to compute the derivative of the inputs
  data   = Variable(data, requires_grad=False)
  target = Variable(target.long(), requires_grad=False)

  # set gradient to 0
  optimizer.zero_grad()
  # feed forward
  pred = model(data)
  # loss calculation
  loss = criterium(pred, target.view(-1))
  
  print('Loss {:.4f} at iter {:d}'.format(loss.item(), k))
  model.print_weights()

  # predictions
  pred = pred.exp().detach()     # exp of the log prob = probability
  _, index = torch.max(pred,1)   # index of the class with maximum probability
  pred = pred.numpy()
  index = index.numpy()

  # calculate auc-roc score
  pred_true = target
  pred_scores = np.array(pred[:, 1])
  auc_score = metrics.roc_auc_score(pred_true, pred_scores)
  print('auc score: {:.3f}'.format(auc_score))

