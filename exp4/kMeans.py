# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:41:02 2018

@author: Administrator
"""

import numpy as np

def loadDataSet(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    dataMat = np.zeros((numberOfLines,2))
    index = 0
    # 样本数据：1.658985	4.285136
    for line in arrayOfLines:
        templine = line.strip().split('\t')
        dataMat[index,0] = float(templine[0])
        dataMat[index,1] = float(templine[1])
        index += 1
    
    return dataMat

def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for i in range(n):
        minJ = np.min(dataSet[:,i]) 
        rangeJ = float(np.max(dataSet[:,i]) - minJ)
        centroids[:,i] = minJ + rangeJ * np.random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = np.shape(dataSet)[0]
    # 使用一个矩阵辅助记录，第一列保存所属质心下标，
    # 第二列保存到该质心的距离的平方
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for temp in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == temp)[0]] # .A 转化为数组
            centroids[temp,:] = np.mean(ptsInClust,axis=0)
            
    return centroids,clusterAssment
                
if __name__=="__main__":
    dataMat = np.mat(loadDataSet("./testSet.txt"))
    centroids,clusterAssment = kMeans(dataMat,4)
                
    
    