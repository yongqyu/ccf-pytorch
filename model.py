import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleMF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(SimpleMF, self).__init__()
        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)

    def forward(self, u, v):
        u = self.u_emb(u)
        v = self.v_emb(v)
        return F.sigmoid(torch.sum(torch.mul(u, v), 1)) * 4 + 1

class NMF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, hidden_dim):
        super(SimpleMF, self).__init__()

        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        self.linear1 = nn.Linear(2*dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, u, v):
        u = self.u_emb(u)
        v = self.v_emb(v)
        x = torch.stack((u, v), 1)

        h = nn.ReLU(self.linear1(x))
        h = self.linear2(h)
        return F.sigmoid(h) * 4 + 1

class MFC(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, hidden_dim):
        super(SimpleMF, self).__init__()

        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        self.conv = nn.Conv1d(2, hidden_dim, 3)
        self.linser = nn.Linear()

    def forward(self, u, v):
        u = self.u_emb(u)
        v = self.v_emb(v)
        x = torch.stack((u, v), 1)
        print(x.size())
        return F.sigmoid(x) * 4 + 1
