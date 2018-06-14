from __future__ import division, print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FactorAnalysis, FastICA, PCA, NMF, LatentDirichletAllocation

def init_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def setup_logger(logger_name, log_file, level = logging.INFO, resume=False):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a' if resume else 'w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)
    return l

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)

def show_config(config):
    print('========== Training Arguments ==========')
    for key in config.keys():
        print('  %s: %s' % (key, str(config[key])))
    print('========================================')

# Feature Extraction
def FA(data, dim):
    fa = FactorAnalysis(n_components=dim)
    fa.fit(data)
    return fa.transform(data)

def ICA(data, dim):
    ica = FastICA(n_components=dim)
    ica.fit(data)
    return ica.transform(data)

def skPCA(data, dim):
    model = PCA(n_components=dim)
    model.fit(data)
    return model.transform(data)

def skNMF(data, dim):
    model = NMF(n_components=dim)
    model.fit(data)
    return model.transform(data)

# Max-min norm
def max_min(data):
    model = MinMaxScaler()
    model.fit(data)
    return model.transform(data)

if __name__ == "__main__":
    print(latest_model("trained_models", "drop_connect"))