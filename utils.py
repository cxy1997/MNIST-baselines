from __future__ import division, print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging

def init_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def setup_logger(logger_name, log_file, level = logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fileHandler = logging.FileHandler(log_file, mode = 'w')
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

def latest_model(model_dir, model_name):
    models = os.listdir(os.path.join(model_dir, model_name))
    model = sorted(models, key=lambda x: int(x[6:-4]))[-1]
    return os.path.join(model_dir, model_name, model), int(model[6:-4])

if __name__ == "__main__":
    print(latest_model("trained_models", "drop_connect"))