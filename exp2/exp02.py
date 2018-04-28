#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:34:24 2018

@author: ronething
"""

from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 			['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                			['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 			['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 			['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 			['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1表示侮辱性0表示正常
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word:%s is not in my Vocabulary!'% word)
    
    return returnVec #返回全为0和1的向量

def trainNB1(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWord=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/len(trainCategory)
    p0Num=ones(numWord);p1Num=ones(numWord)# 初始化为1
    p0Demon=2;p1Demon=2 #初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i]==0:
            p0Num+=trainMatrix[i]
            p0Demon+=sum(trainMatrix[i])
        else:
            p1Num+=trainMatrix[i]
            p1Demon+=sum(trainMatrix[i])
    p0Vec=log(p0Num/p0Demon) #对结果求对数
    p1Vec=log(p1Num/p1Demon) #对结果求自然对数
    return p0Vec,p1Vec,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
    
    
def testingNB():
    listPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB1(trainMat,listClasses)
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    
if __name__=="__main__":
    testingNB()