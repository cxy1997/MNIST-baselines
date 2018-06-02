# data loader for mnist dataset
# the path is based on root dictory of mnist_project

import os
import numpy as np
from PIL import Image

class MnistLoader(object):
    DATA_NUM = (60000, 10000)       # The number of figures
    FIG_W = 45                      # width of each figure

    def __init__(self, flatten=False, data_path='data'):
        '''
        :param data_path: the path to mnist dataset
        '''

        self.data_train = np.concatenate([np.load(os.path.join(data_path, 'mnist_train', 'mnist_train_data_part1.npy')), np.load(os.path.join(data_path, 'mnist_train', 'mnist_train_data_part2.npy'))], axis=0)
        self.label_train = np.load(os.path.join(data_path, 'mnist_train', 'mnist_train_label.npy'))
        self.data_test = np.load(os.path.join(data_path, 'mnist_test', 'mnist_test_data.npy'))
        self.label_test = np.load(os.path.join(data_path, 'mnist_test', 'mnist_test_label.npy'))

        if flatten:
            self.data_train = self.data_train.reshape(self.data_train.shape[0], -1)
            self.data_test = self.data_test.reshape(self.data_test.shape[0], -1)

    def demo(self):
        # show the structure of data & label
        print('Train data:', self.data_train.shape)
        print('Train labels:', self.label_train.shape)
        print('Test data:', self.data_test.shape)
        print('Test labels:', self.label_test.shape)

        # choose a random index
        ind = np.random.randint(0, self.DATA_NUM[0])

        # print the index and label
        print("index: ", ind)
        print("label: ", self.label_train[ind])

        # save the figure
        im = Image.fromarray(self.data_train[ind].reshape(self.FIG_W, self.FIG_W))
        im.show()
        # im.save("demo.png")

if __name__ == '__main__':
    loader = MnistLoader()
    loader.demo()