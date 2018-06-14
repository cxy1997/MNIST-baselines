from sklearn.naive_bayes import GaussianNB

import sys
sys.path.append('/home/shin/mlp')

from data_loader import MnistLoader

def NB():
    loader = MnistLoader(flatten=True, data_path='../data', var_per=None)
    model = GaussianNB()

    model.fit(loader.data_train, loader.label_train)
    print('model trained')
    res = model.score(loader.data_test, loader.label_test)
    print(res)

    return res

if __name__ == '__main__':
    NB()