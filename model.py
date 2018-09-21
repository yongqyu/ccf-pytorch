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
        linear = []
        for (in_d, out_d) in zip(layers[:-1], layers[1:]):
            linear.append(nn.Linear(in_d, out_d))
            linear.append(nn.ReLU())
        self.linear = nn.Sequential(*linear)
        self.out_lin = nn.Linear(layers[-1], 1)

    def forward(self, u, v):
        u = self.u_emb(u)
        v = self.v_emb(v)
        x = torch.cat((u, v), 1)

        h = self.linear(x)
        h = self.out_lin(h).squeeze(-1)
        return h

class MFC(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, hidden_dim):
        super(MFC, self).__init__()

        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        self.conv1 = nn.Conv1d(2, hidden_dim, 3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3)
        self.linear = nn.Linear(16*3, 1)

    def forward(self, u, v):
        u = self.u_emb(u)
        v = self.v_emb(v)
        x = torch.stack((u, v), 1)

        h = F.max_pool1d(self.conv1(x), kernel_size=2)
        h = F.max_pool1d(self.conv2(h), kernel_size=2)
        h = self.linear(h.view(h.size(0), -1)).squeeze(-1)
        return F.sigmoid(h) * 4 + 1
