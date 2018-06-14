from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from utils import weights_init

class dnn(nn.Module):
    def __init__(self, in_features=2025, classes=10, activation='relu', batch_norm=False):
        super(dnn, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm

        if batch_norm:
            self.fc1 = nn.Linear(in_features, 500)
            self.bn1 = nn.BatchNorm1d(500)
            self.fc2 = nn.Linear(500, 200)
            self.bn2 = nn.BatchNorm1d(200)
            self.fc3 = nn.Linear(200, 50)
            self.bn3 = nn.BatchNorm1d(50)
            self.fc4 = nn.Linear(50, classes)
        else:
            self.fc1 = nn.Linear(in_features, 500)
            self.fc2 = nn.Linear(500, 200)
            self.fc3 = nn.Linear(200, 50)
            self.fc4 = nn.Linear(50, classes)

    def forward(self, x):
        if self.activation == 'relu':
            if self.batch_norm:
                x = F.relu(self.bn1(self.fc1(x)))
                x = F.relu(self.bn2(self.fc2(x)))
                x = F.relu(self.bn3(self.fc3(x)))
            else:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))

        elif self.activation == 'tanh':
            if self.batch_norm:
                x = F.tanh(self.bn1(self.fc1(x)))
                x = F.tanh(self.bn2(self.fc2(x)))
                x = F.tanh(self.bn3(self.fc3(x)))
            else:
                x = F.tanh(self.fc1(x))
                x = F.tanh(self.fc2(x))
                x = F.tanh(self.fc3(x))

        return self.fc4(x)