from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.append('../')

from data_loader import MnistLoader

def KNN():
    loader = MnistLoader(flatten=True, data_path='../data', var_per=None)
    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(loader.data_train, loader.label_train)
    print('model trained')
    res = model.score(loader.data_test, loader.label_test)
    print(res)

    return res

if __name__ == '__main__':
    KNN()