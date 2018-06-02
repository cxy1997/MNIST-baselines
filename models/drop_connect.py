from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import weights_init

class drop_connect_layer(nn.Module):
    def __init__(self, in_channels, out_channels, prob=0.5, bias=True):
        super(drop_connect_layer, self).__init__()

        self.weight = Variable(torch.zeros(out_channels, in_channels), requires_grad=True)
        w_bound = np.sqrt(6. / (out_channels + in_channels))
        self.weight.data.uniform_(-w_bound, w_bound)
        self.weight_dropout = nn.Dropout(p=prob)

        if bias:
            self.bias = Variable(torch.zeros(out_channels), requires_grad=True)
            self.bias.data.fill_(0)
            self.bias_dropout = nn.Dropout(p=prob)
        else:
            self.bias = None

    def forward(self, x):
        weight = self.weight_dropout(self.weight)
        bias = self.bias_dropout(self.bias) if self.bias is not None else None
        return F.linear(x, weight, bias)

class drop_connect_net(nn.Module):
    def __init__(self, in_channels=2025, classes=10, prob=0.5, bias=True):
        super(drop_connect_net, self).__init__()

        self.dc1 = drop_connect_layer(in_channels, 600, prob=prob, bias=bias)
        self.dc2 = drop_connect_layer(600, 600, prob=prob, bias=bias)
        self.dc3 = drop_connect_layer(600, classes, prob=prob, bias=bias)

    def forward(self, x):
        x = F.relu(self.dc1(x))
        x = F.relu(self.dc2(x))
        return F.relu(self.dc3(x))