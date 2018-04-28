# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:50:19 2018

@author: Administrator
"""

import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def kNNClassify(newInput,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #求出行数
    diff = np.tile(newInput,(dataSetSize,1))-dataSet #求出新向量与原先矩阵中向量的差
    diff2 = diff**2 # 求出差的平方
    diffsum = np.sum(diff2,axis=1)# 算出向量平方的和
    diffsum2 = diffsum**0.5#算出欧式距离
    #print(diffsum2)
    sortedDistIndices = np.argsort(diffsum2) # argsort()函数的返回值为按照升序排序的下标
    #print(sortedDistIndices)
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    # 选择k个最相似数据中出现次数最多的分类
    '''
    maxcount = 0
    maxindex = ""
    for key,value in classCount.items():
        if value > maxcount:
            maxcount=value
            maxindex=key
    predictedClass={}
    predictedClass[maxindex]=maxcount
    return predictedClass
    '''
    predictedClass = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return predictedClass[0][0]

if __name__=='__main__':
    group,labels = createDataSet()
    predictedClass = kNNClassify([0,0],group,labels,3)
    print(predictedClass)