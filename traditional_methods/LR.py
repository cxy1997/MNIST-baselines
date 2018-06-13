from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('../')

from data_loader import MnistLoader

def LR():
    loader = MnistLoader(flatten=True, data_path='../data', var_per=None)
    model = LogisticRegression(penalty='l2')

    model.fit(loader.data_train, loader.label_train)
    print('model trained')
    res = model.score(loader.data_test, loader.label_test)
    print(res)

    return res

if __name__ == '__main__':
    LR()