from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.drop_connect import drop_connect_layer

if __name__ == '__main__':
    x = Variable(torch.ones(5, 3))
    print(x)
    f = drop_connect_layer(3, 5)
    y = f(x)
    print(y)