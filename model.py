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
    def __init__(self, num_user, num_item, emb_dim, layers):
        super(NMF, self).__init__()

        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        linears = []
        for (in_d, out_d) in zip(layers[:-1], layers[1:]):
            linears.append(nn.Linear(in_d, out_d))
            linears.append(nn.ReLU())
        self.linear = nn.Sequential(*linears)
        self.out_lin = nn.Linear(layers[-1], 1)

    def forward(self, u, v, n):
        u = self.u_emb(u)
        v = self.v_emb(v)
        n = self.v_emb(n)
        x = torch.cat((u, v), 1)
        n_x=torch.cat((u.unsqueeze(1).expand_as(n), n), 2)
        n_x=n_x.view(-1,n_x.size(-1))

        h = self.linear(x)
        h = self.out_lin(h).squeeze(-1)
        n_h=self.linear(n_x)
        n_h=self.out_lin(n_h).squeeze(-1)
        return F.sigmoid(h), F.sigmoid(n_h)

class MFC(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, layers):
        super(MFC, self).__init__()

        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        convs = []
        for (in_d, out_d) in zip(layers[:-1], layers[1:]):
            convs.append(nn.Conv1d(in_d, out_d, 2))
            convs.append(nn.MaxPool1d(2))
        self.conv = nn.Sequential(*convs)
        self.linear = nn.Linear(layers[-1], 1)

    def forward(self, u, v, n):
        u = self.u_emb(u)
        v = self.v_emb(v)
        n = self.v_emb(n.view(-1))
        x = torch.stack((u, v), 1)
        n = torch.stack((u, n), 1)

        h = self.conv(x)
        h = self.linear(h.view(h.size(0), -1)).squeeze(-1)
        return h
