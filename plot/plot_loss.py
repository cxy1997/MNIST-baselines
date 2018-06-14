# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import copy
import matplotlib.pyplot as plt

models = {'vgg':           'VGG-19',
          'resnet':        'ResNet-101',
          'preact_resnet': 'PreActResNet-152',
          'googlenet':     'GoogLeNet',
          'densenet':      'DenseNet-161',
          'resnext':       'ResNeXt-29-8x64d',
          'mobilenet':     'MobileNet',
          'mobilenetv2':   'MobileNetV2',
          'dpn':           'DPN-92',
          'senet':         'SENet-18',
          'shufflenet':    'ShuffleNetG3',
          'capsnet':       'CapsNet',
          'pnasnet':       'PNASNet',
          'lenet':         'LeNet'}

def smooth(array, m=2):
    _array = copy.deepcopy(array)
    n = _array.shape[0]
    for i in range(1, n):
        _array[i] = np.mean(array[max(0, i - m): min(n, i + m + 1)])
    return _array

def plot_single(method):
    epoch = []
    acc = []
    with open('logs/%s.log' % method, 'r') as f:
        for line in f.readlines():
            entry = line.split(' ')
            epoch.append(eval(entry[3][:-1]))
            acc.append(100*eval(entry[5][:-1]))
    epoch = np.array(epoch)
    acc = np.array(acc)
    epoch = smooth(epoch)
    acc = smooth(acc)
    plt.plot(epoch, acc)

def plot_all():
    plt.figure(figsize = (8, 5))
    legend = []
    for key in models.keys():
        plot_single(key)
        legend.append(models[key])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim((0, 5000))
    plt.legend(legend)
    plt.tight_layout()
    plt.savefig('logs/figures/loss.png', dpi = 300)


if __name__ == '__main__':
    plot_all()