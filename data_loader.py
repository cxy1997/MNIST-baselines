# data loader for mnist dataset
# the path is based on root dictory of this repo

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class MnistLoader(object):
    DATA_SIZE = (60000, 10000)      # number of figures
    FIG_W = 45                      # width of each figure
    CLASSES = 10                    # number of classes

    def __init__(self, flatten=False, data_path='data'):
        '''
        :param data_path: the path to mnist dataset
        '''
        self.flatten = flatten
        self._load(data_path)

        # normalize the data
        self.data_train = (self.data_train - self.data_train.mean()) / self.data_train.std()
        self.data_test = (self.data_test - self.data_test.mean()) / self.data_test.std()

    # load data according to different configurations
    def _load(self, data_path='data'):
        self.data_train = np.concatenate([np.load(os.path.join(data_path, 'mnist_train', 'mnist_train_data_part1.npy')), np.load(os.path.join(data_path, 'mnist_train', 'mnist_train_data_part2.npy'))], axis=0).astype(np.float32)
        self.label_train = np.load(os.path.join(data_path, 'mnist_train', 'mnist_train_label.npy')).astype(np.int64)
        self.data_test = np.load(os.path.join(data_path, 'mnist_test', 'mnist_test_data.npy')).astype(np.float32)
        self.label_test = np.load(os.path.join(data_path, 'mnist_test', 'mnist_test_label.npy')).astype(np.int64)

        # flatten data
        if self.flatten:
            self.data_train = self.data_train.reshape(self.data_train.shape[0], -1)
            self.data_test = self.data_test.reshape(self.data_test.shape[0], -1)

    def demo(self):
        # show the structure of data & label
        print('Train data:', self.data_train.shape)
        print('Train labels:', self.label_train.shape)
        print('Test data:', self.data_test.shape)
        print('Test labels:', self.label_test.shape)

        # choose a random index
        ind = np.random.randint(0, self.DATA_SIZE[0])

        # print the index and label
        print("index: ", ind)
        print("label: ", self.label_train[ind])

        # save the figure
        plt.imshow(self.data_train[ind].reshape(self.FIG_W, self.FIG_W))
        plt.show()
        im.save("demo.png")