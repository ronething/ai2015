# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 22:10:10 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

# 数据读取函数
def loadDataSet(filename):
    fr = open(filename)
    stringArr = [line.strip().split('\t') for line in \
                 fr.readlines()]
    dataArr = [list(map(float,line)) for line in stringArr]
    fr.close()
    
    return np.mat(dataArr)

# 根据特征值和比例确定主成分个数
def getNumOfFeat(eigVals,percentage):
    # 使用Numpy中的sort函数对特征值进行排序
    # sort()是升序排序，需要倒过来
    # 将排好序的特征值保存到变量sortArray中
    eigVals.sort()
    sortArray = eigVals[::-1]
    arraySum = np.sum(sortArray)
    tempSum = 0 #用以保存主成分的方差之和
    numOfFeat = 0 #用以保存主成分个数
    for i in sortArray:
        tempSum += i
        numOfFeat += 1
        if tempSum >= arraySum*percentage:
            return numOfFeat
    
# pca函数
def pca(dataMat,percentage=0.9):
    # 按列计算均值，保存到meanVals
    # 使用numpy的mean函数
    # 注意参数axis的取值
    meanVals = np.mean(dataMat,axis=0)
    # 去平均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵
    # rowvar=0表示数据集中一行代表一条数据
    covMat = np.cov(meanRemoved,rowvar=0)
    # 计算协方差矩阵的特征值和特征向量
    # 使用numpy中linalg模块的eig()函数
    # 该函数有两个返回值，第一个是特征值，第二个是特征向量
    # 注意先用numpy的mat函数变换一下协方差矩阵
    # 要求特征值用变量eigVals，特征向量用变量eigVects
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    # 调用上一步的函数获取主成分个数，保存到k
    k = getNumOfFeat(eigVals,percentage)
    # 对特征值排序，获取最大的k个特征值的下标
    eigValInd = np.argsort(eigVals)
    """
    emm 例如a=[3,4,1,2]
    b=np.argsort(a) 则b=[2,3,0,1]
    b[:-2:-1]=[1]
    可以这么理解：b[::-1]=[1,0,3,2]
    b[-2]=0
    所以b[:-2:-1]=[1]
    """
    eigValInd = eigValInd[:-(k+1):-1]
    # 根据k个特征值的下标获取对应的特征向量
    redEigVects = eigVects[:,eigValInd]
    # 根据k个特征向量将取均值化的数据转换到新空间
    # 直接相乘就好，保存到lowDDataMat
    lowDDataMat = meanRemoved*redEigVects
    # 根据降维后的数据重构回原来的数据空间中
    reconMat = (lowDDataMat*redEigVects.T)+meanVals
    
    return lowDDataMat,reconMat

# 测试函数
def test(percentage=0.9):
    dataMat = loadDataSet("./testSet.txt")
    lowDMat,reconMat = pca(dataMat,percentage)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],\
               dataMat[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(reconMat[:,0].flatten().A[0],\
               reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
    plt.show()
    print('Low dimension shape is',np.shape(lowDMat),\
              'original shape is',np.shape(reconMat))
    
if __name__=="__main__":
    test(0.5)
