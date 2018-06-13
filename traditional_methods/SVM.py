from sklearn.svm import SVC

import sys
sys.path.append('../')

from data_loader import MnistLoader

def SVM():
    loader = MnistLoader(flatten=True, data_path='../data', var_per=None)
    model = SVC(kernel='rbf')

    model.fit(loader.data_train, loader.label_train)
    print('model trained')
    res = model.score(loader.data_test, loader.label_test)
    print(res)

    return res

if __name__ == '__main__':
    SVM()