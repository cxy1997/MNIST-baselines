#-*-coding:utf8-*-

import numpy as np

# zero mean
def zeroMean(dataMat):
    meanVal = np.mean(dataMat,axis=0)     # get the mean value of each feature
    newData = dataMat - meanVal
    return newData, meanVal

def percentage2n(eigVals,percentage):
    sortArray = np.sort(eigVals)   # ascending order
    sortArray = sortArray[-1::-1]  # descending order
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num

def pca(dataMat,percentage=0.99):
    dataMat,meanVal=zeroMean(dataMat)
    # print "datamat type :" + str(type(dataMat))

    print ("Now computing covariance matrix...")
    covMat=np.cov(dataMat,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    # print "covmat type :" + str(type(covMat))

    print ("Finished. Now solve eigen values and vectors...")
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    # print "eigVals type :" + str(type(eigVals))
    # print "eigVects type :" + str(type(eigVects))

    print ("Finished. Now select eigen vectors...")
    n=percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    # print "eigValIndice type :" + str(type(eigValIndice))

    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    print ("Finished. Now generating new data...")
    # print "n_eigVect type :" + str(type(n_eigVect))

    lowDDataMat=dataMat*n_eigVect               #低维特征空间的数据
    # reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据
    return np.array(lowDDataMat)