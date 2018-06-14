from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from utils import weights_init

class dnn(nn.Module):
    def __init__(self, in_features=2025, classes=10, activation='relu'):
        super(dnn, self).__init__()
        self.activation = activation

        self.fc1 = nn.Linear(in_features, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, classes)

    def forward(self, x):
        if self.activation == 'relu':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        elif self.activation == 'tanh':
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            x = F.tanh(self.fc3(x))

        return self.fc4(x)