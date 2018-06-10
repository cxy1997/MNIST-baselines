from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from utils import weights_init

class drop_connect_layer(nn.Module):
    def __init__(self, in_features, out_features, prob=0.5, bias=True):
        super(drop_connect_layer, self).__init__()

        self.weight = Parameter(torch.zeros(out_features, in_features), requires_grad=True)
        w_bound = np.sqrt(6. / (out_features + in_features))
        self.weight.data.uniform_(-w_bound, w_bound)
        self.weight_dropout = nn.Dropout(p=prob)

        if bias:
            self.bias = Parameter(torch.zeros(out_features), requires_grad=True)
            self.bias.data.fill_(0)
            self.bias_dropout = nn.Dropout(p=prob)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight_dropout(self.weight)
        bias = self.bias_dropout(self.bias) if self.bias is not None else None
        return F.linear(x, weight, bias)

class drop_connect_net(nn.Module):
    def __init__(self, in_features=2025, classes=10, prob=0.5, bias=True):
        super(drop_connect_net, self).__init__()

        self.dc1 = nn.Linear(in_features, 600)#drop_connect_layer(in_features, 2000, prob=prob, bias=bias)
        self.dc2 = nn.Linear(600, classes)#drop_connect_layer(20, classes, prob=prob, bias=bias)

    def forward(self, x):
        x = F.relu(self.dc1(x))
        return self.dc2(x)