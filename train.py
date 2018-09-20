# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

from model import *
from config import get_args
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# UserID::MovieID::Rating::Timestamp (5-star scale)
# Importing the dataset
train = pd.read_pickle('./data/train.pkl')
val   = pd.read_pickle('./data/val.pkl')
test  = pd.read_pickle('./data/test.pkl')

train_loader = DataLoader(TensorDataset(torch.tensor(train.values)), args.batch_size, args.data_shuffle)
val_loader   = DataLoader(TensorDataset(torch.tensor(val.values)), args.batch_size, args.data_shuffle)
test_loader  = DataLoader(TensorDataset(torch.tensor(test.values)), args.batch_size, args.data_shuffle)

# Getting the number of users and movies
num_users = int(max(max(train.values[:,0]), max(val.values[:,0]), max(test.values[:,0]))) + 1
num_movies = int(max(max(train.values[:,1]), max(val.values[:,1]), max(test.values[:,1])))+ 1

# Creating the architecture of the Neural Network
if args.model == 'SimpleMF':
    model = SimpleMF(num_users, num_movies, args.emb_dim)
elif args.model == 'NMF':
    model = NMF(num_users, num_movies, args.emb_dim, args.emb_dim)
elif args.model == 'MFC':
    model = MFC(num_users, num_movies, args.emb_dim, 16)
if torch.cuda.is_available():
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr = 0.01, weight_decay = 0.5)

best_epoch = 0
best_loss  = 9999.


def train():
    # Training
    for epoch in range(args.num_epochs):
        train_loss = 0
        model.train()
        for s, x in enumerate(train_loader):
            x = x[0].to(device)
            u = Variable(x[:,0])
            v = Variable(x[:,1])
            r = Variable(x[:,2]).float()

            pred = model(u, v)
            loss = criterion(pred, r)
            train_loss += np.sqrt(loss.item())

            model.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for s, x in enumerate(val_loader):
                    x = x[0].to(device)
                    u = Variable(x[:,0])
                    v = Variable(x[:,1])
                    r = Variable(x[:,2]).float()

                    pred = model(u, v)
                    loss = criterion(r, pred)
                    val_loss += np.sqrt(loss.item())
            print('[val loss] : '+str(val_loss/s))
            if best_loss > (val_loss/s):
                best_loss = (val_loss/s)
                best_epoch= epoch
            torch.save(model,
                       os.path.joint(args.model_path+args.model,
                       'model-%d.pkl'%(epoch+1)))

def test():
    # Test
    model.load_state_dict(torch.load(os.path.joint(args.model_path+args.model,
                          'model-%d.pkl'%(best_epoch+1))))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for s, x in enumerate(test_loader):
            x = x[0].to(device)
            u = Variable(x[:,0])
            v = Variable(x[:,1])
            r = Variable(x[:,2]).float()

            pred = model(u, v)
            loss = criterion(r, pred)
            test_loss += np.sqrt(loss.item())

    print('[test loss] : '+str(test_loss/s))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    if args.mode == 'test':
        best_epoch = args.test_epoch
    test()
