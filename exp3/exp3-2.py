# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:50:19 2018

@author: Administrator
"""

import numpy as np
import os
import operator

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fileIn = open(filename)
    for i in range(32):
        lineStr = fileIn.readline()
        for j in range(32):
            returnVect[0,i*32+j] = int(lineStr[j])
    
    fileIn.close()
    return returnVect

def loadDataSet():
    print("---Getting training set...")
    
    dataSetDir = "J:/Files/homework/人工智能实验/实验三/"
    trainingFileList = os.listdir(dataSetDir+"trainingDigits")
    numSamples = len(trainingFileList)
    
    train_x = np.zeros((numSamples,1024))
    train_y = []
    for i in range(numSamples):
        filename = trainingFileList[i]
        train_x[i,:] = img2vector(dataSetDir+"trainingDigits/{0}".format(filename))
        label = int(filename.split('_')[0])
        train_y.append(label)
        
    print("---Getting test set...")
    testFileList = os.listdir(dataSetDir+"testDigits")
    numSamples = len(testFileList)
    
    test_x = np.zeros((numSamples,1024))
    test_y = []
    for i in range(numSamples):
        filename = testFileList[i]
        test_x[i,:] = img2vector(dataSetDir+"testDigits/{0}".format(filename))
        label = int(filename.split('_')[0])
        test_y.append(label)
    
    return train_x,train_y,test_x,test_y

def kNNClassify(newInput,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #求出行数
    diff = np.tile(newInput,(dataSetSize,1))-dataSet #求出新向量与原先矩阵中向量的差
    diff2 = diff**2 # 求出差的平方
    diffsum = np.sum(diff2,axis=1)# 算出向量平方的和
    diffsum2 = diffsum**0.5#算出欧式距离
    sortedDistIndices = np.argsort(diffsum2)
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    # 选择k个最相似数据中出现次数最多的分类
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def testHandWritingClass():
    print("Step 1: Load data...")
    train_x,train_y,test_x,test_y=loadDataSet()
    print("Step 2: Training...")
    pass
    print("Step 3: Testing...")
    numTestSamples = test_x.shape[0]
    matchCount = 0
    for i in range(numTestSamples):
        testresult = kNNClassify(test_x[i],train_x,train_y,3)
        #print ("第",i+1,"组：","预测值:",testresult,"真实值:",test_y[i])
        if testresult != test_y[i]:
            print ("识别错误：第",i+1,"组：","预测值:",testresult,"真实值:",test_y[i])
        else:
            matchCount+=1
    accuracy = np.float(matchCount/numTestSamples)
    print("Step 4: Show the result...")
    print("The classify accuracy is %.2f%%" % (accuracy * 100))
    
if __name__=='__main__':
    testHandWritingClass()