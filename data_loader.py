# data loader for mnist dataset
# the path is based on root dictory of mnist_project

import numpy as np
from PIL import Image

class MnistLoader(object):
    DATA_NUM = (60000, 10000)       # The number of figures
    FIG_W = 45                      # width of each figure

    def __init__(self, fix=True, path='./data/'):
        '''
        :param path: the path to mnist dataset
        '''

        self.data_train, self.label_train = self._load(path + 'mnist_train/mnist_train_', 0)
        self.data_test, self.label_test = self._load(path + 'mnist_test/mnist_test_', 1)

        if fix:
            self._fix()

    def _load(self, path, mode=0):
        '''
        :param path: data path
        :param mode: 0 - train; 1 - test
        '''

        # load from files
        data = np.fromfile(path + 'data', dtype=np.uint8)
        label = np.fromfile(path + 'label', dtype=np.uint8)

        # reshape the matrix
        data = data.reshape(self.DATA_NUM[mode], self.FIG_W, self.FIG_W)

        return data, label

    def _fix(self):
        # flatten the data
        self.data_train = self.data_train.reshape(self.data_train.shape[0], self.FIG_W * self.FIG_W)
        self.data_test = self.data_test.reshape(self.data_test.shape[0], self.FIG_W * self.FIG_W)

        # change label into one-hot vector
        self.label_train = np.eye(10)[self.label_train.reshape(-1)]
        self.label_test = np.eye(10)[self.label_test.reshape(-1)]

    def demo(self):
        # show the structure of data & label
        print(self.data_train.shape)
        print(self.label_train.shape)

        # choose a random index
        ind = np.random.randint(0, self.DATA_NUM[0])

        # print the index and label
        print("index: ", ind)
        print("label: ", self.label_train[ind])

        # save the figure
        im = Image.fromarray(self.data_train[ind].reshape(self.FIG_W, self.FIG_W))
        im.show()
        im.save("demo.png")

if __name__ == '__main__':
    loader = MnistLoader()
    loader.demo()