#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 20:03:55 2018

@author: ronething
"""

from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
      			[1, 1, 'yes'],
      			[1, 0, 'no'],
      			[0, 1, 'no'],
      			[0, 1, 'no']]
    labels = ['no surfacing','flippers']
    
    return dataSet,labels

def calcEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    entropy = 0.0
    for key in labelCounts.keys():
        pxi = float(labelCounts[key])/numEntries
        entropy -= pxi*log(pxi,2)
        
    return entropy

def splitDataSet(dataSet,axis,value):
    returnDataSet = []
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            returnDataSet.append(reducedFeatVec)
    
    return returnDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 #获取属性个数
    baseEntropy = calcEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 获取数据集中某一属性的所有取值
        uniqueVals = set(featList) # 获取该属性所有不重复的取值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            pxi = len(subDataSet)/float(len(dataSet))
            # 特征A对数据集D的信息增益公式实现
            newEntropy += pxi*calcEntropy(subDataSet)
        
        infoGain = baseEntropy - newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),\
                              key=operator.itemgetter(1),reverse=True)
    
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] #获取类别列表
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(dataSet)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    
    myTree = {bestFeatLabel:{}}
    
    del(labels[bestFeat])
    
    featList = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featList)
    for value in uniqueVals:
        subLabel = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabel)
    
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__=="__main__":
    data,labels = createDataSet()
    templabels = labels[:]
    # 这里很神奇 如果用labels传入createTree方法 最后labels会被删除'no surfacing' key 值
    myTree = createTree(data,templabels)
    print(classify(myTree,labels,[1,0]))