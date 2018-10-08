import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class GMF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(GMF, self).__init__()
        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)

    def forward(self, u, v):
        u = self.u_emb(u)
        v = self.v_emb(v)
        return torch.mul(u, v)

class NCF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, layers):
        super(NCF, self).__init__()

        self.gmf = GMF(num_user, num_item, emb_dim)
        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        linears = []
        for (in_d, out_d) in zip(layers[:-1], layers[1:]):
            linears.append(nn.Linear(in_d, out_d))
            linears.append(nn.ReLU())
            linears.append(nn.Dropout(p=0.2))
        self.linear = nn.Sequential(*linears)
        self.predict= nn.Linear(emb_dim+layers[-1], 1)

    def forward(self, u, v, n):
        # GMF
        gmf = self.gmf(u,v)
        gmf_n=self.gmf(u.unsqueeze(1).expand_as(n),n).view(-1,gmf.size(-1))

        # MLP
        u = self.u_emb(u)
        v = self.v_emb(v)
        n = self.v_emb(n)
        x = torch.cat((u, v), 1)
        x_n=torch.cat((u.unsqueeze(1).expand_as(n), n), 2)
        x_n=x_n.view(-1,x_n.size(-1))

        h = self.linear(x)
        h_n=self.linear(x_n)

        # Fusion
        pred = self.predict(torch.cat((gmf,h), 1)).view(-1)
        pred_n=self.predict(torch.cat((gmf_n,h_n), 1)).view(-1)

        return pred, pred_n

class ONCF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, layers):
        super(ONCF, self).__init__()

        self.gmf = GMF(num_user, num_item, emb_dim)
        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        convs = []
        for (in_d, out_d) in zip(layers[:-1], layers[1:]):
            convs.append(nn.Conv2d(in_d, out_d, 2, 2))
            convs.append(nn.ReLU())
        self.conv = nn.Sequential(*convs)
        self.predict= nn.Linear(emb_dim+layers[-1], 1)

    def forward(self, u, v, n):
        # GMF
        gmf = self.gmf(u,v)
        gmf_n=self.gmf(u.unsqueeze(1).expand_as(n),n).view(-1,gmf.size(-1))

        # MLP
        u = self.u_emb(u)
        v = self.v_emb(v)
        n = self.v_emb(n)
        x = torch.bmm(u.unsqueeze(2), v.unsqueeze(1))
        x_n=torch.bmm(u.repeat(1,n.size(1)).view(-1,n.size(-1),1),
                      n.view(-1,1,n.size(-1)))

        h = self.conv(x.unsqueeze(1)).squeeze()
        h_n=self.conv(x_n.unsqueeze(1)).squeeze()

        # Fusion
        pred = self.predict(torch.cat((gmf,h), 1)).view(-1)
        pred_n=self.predict(torch.cat((gmf_n,h_n), 1)).view(-1)

        return pred, pred_n

class CCF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, layers):
        super(CCF, self).__init__()

        self.gmf = GMF(num_user, num_item, emb_dim)
        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        convs = []
        for (in_d, out_d) in zip(layers[:-1], layers[1:]):
            convs.append(nn.Conv1d(in_d, out_d, 2))
            convs.append(nn.ReLU())
            convs.append(nn.AvgPool1d(2))
        '''
        convs = []
        for i, (in_d, out_d) in enumerate(zip(layers[:-1], layers[1:])):
            convs.append(nn.Conv1d(in_d, out_d, 2, 2))
            #convs.append(nn.BatchNorm1d(in_d))
            convs.append(nn.ReLU())
            #convs.append(nn.Dropout(p=0.5))
        '''
        self.conv = nn.Sequential(*convs)
        self.pred = nn.Linear(emb_dim+layers[-1], 1)

    def forward(self, u, v, n):
        # GMF
        gmf = self.gmf(u,v)
        gmf_n=self.gmf(u.unsqueeze(1).expand_as(n),n).view(-1,gmf.size(-1))

        # CNN
        u = self.u_emb(u)
        v = self.v_emb(v)
        n = self.v_emb(n)
        x = torch.stack((u, v), 1)
        x_n=torch.stack((u.repeat(1,n.size(1)).view(-1,n.size(-1)),
                         n.view(-1,n.size(-1))), 1)

        h = self.conv(x).view(x.size(0), -1)
        h_n=self.conv(x_n).view(x_n.size(0), -1)

        # Fusion
        pred = self.pred(torch.cat((gmf,h), 1)).view(-1)
        pred_n=self.pred(torch.cat((gmf_n,h_n), 1)).view(-1)

        return pred, pred_n
