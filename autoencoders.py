#Importing Libraries
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
#Importing Dataset
mov = pd.read_csv('ml-1m/movies.dat',sep = ': :',header = None, engine = 'python',encoding = 'Latin-1')
users = pd.read_csv('ml-1m/users.dat',sep = ': :',header = None, engine = 'python', encoding ='Latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = ': :',header = None, engine = 'python',encoding = 'Latin-1')
#Training and Test Set
train_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
train_set = np.array(train_set,dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set,dtype = 'int')
#Getting number of users and movies
user_no = int(max(max(train_set[:,0]),max(test_set[:,0])))
movie_no = int(max(max(train_set[:,1]),max(test_set[:,1])))
#Converting data into array
def con(d):
  new_d = []
  for i in range(1,user_no+1):
    mov_id = d[:,1][d[:,0] == i]
    rat = d[:,2][d[:,0] == i]
    ratings = np.zeros(movie_no)
    ratings[mov_id-1] = rat
    new_d.append(list(ratings))
  return new_d
train_set = con(train_set)
test_set = con(test_set)
#Converting to Torch Tensor
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)
#Architecture of Autoencoders
class SAE(nn.Module):
  def __init__(self, ):
    super(SAE,self).__init__()
    #Connections between Vectors and Hidden Layers
    self.r1 = nn.Linear(movie_no, 20)
    self.r2 = nn.Linear(20,10)
    self.r3 = nn.Linear(10,20)
    self.r4 = nn.Linear(20,movie_no)
    self.acti = nn.Sigmoid()
  def forward(self, x):
    x = self.acti(self.r1(x))
    x = self.acti(self.r2(x))
    x = self.acti(self.r3(x))
    x = self.r4(x)
    return x
sae = SAE()
cond = nn.MSELoss()
optimiser = optim.RMSprop(sae.parameters(), lr = 0.0009, weight_decay = 1 )
#Training
epochs = 200
for i in range(1 , epochs+1):
  loss = 0
  s = 0.
  for j in range(user_no):
    o = Variable(train_set[j]).unsqueeze(0)
    t = o.clone()
    if torch.sum(t.data > 0) > 0 :
      out = sae(o)
      t.require_grad = False
      out[t == 0] = 0
      lo = cond(out,t)
      mean_corrector = movie_no/float(torch.sum(t.data>0) + 1e-10)
      lo.backward()
      loss += np.sqrt(lo.data*mean_corrector)
      s += 1.
      optimiser.step()
  print('Epochs: '+str(i)+' loss:'+str(loss/s))
#Testing
test_loss = 0
s = 0.
for j in range(user_no):
  o = Variable(train_set[j]).unsqueeze(0)
  t = Variable(test_set[j])
  if torch.sum(t.data > 0) > 0 :
    out = sae(o)
    t.require_grad = False
    out[t == 0] = 0
    lo = cond(out,t)
    mean_corrector = movie_no/float(torch.sum(t.data>0) + 1e-10)
    test_loss += np.sqrt(lo.data*mean_corrector)
    s += 1.
print('Loss: '+str(loss/s))
  
      
      
      
