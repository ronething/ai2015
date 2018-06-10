# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 16:41:06 2018

@author: Administrator
"""

# 读数据，这里使用一个简单数据集
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

# 生成单个物品项集
def createC1(dataSet):
    C1 = []
    # 循环查看每一条交易记录
    # 对每一条交易记录，判断记录中的物品是否已经在C1中
    # 如果不在则添加
    # 使用C1.append([XXX])，注意要使用中括号
    for i in dataSet:
        for j in i:
            if [j] not in C1:
                C1.append([j])
    C1.sort()
    return list(map(frozenset,C1))

# 数据集扫描函数
def scanD(D, Ck, minSupport):
    # 创建一个字典保存各个候选集的计数
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # 获取总的交易数
    numItems = float(len(D))
    # 创建返回列表保存频繁项集
    retList = []
    # 创建支持度字典保存各个项集的支持度，以便后续计算置信度
    supportData = {}
    for key in ssCnt:
        tmp = float(ssCnt[key]*1.0/numItems)
        # 如果其支持度不低于(>=)最小支持度，则保留该项集
        if tmp>=minSupport:
            retList.append(key)
        supportData[key] = tmp
    
    return retList,supportData

# 构建一个由k个项组成的候选集列表。这里的Lk是已知的每一项包
# 含 k-1 个物品的频繁项集列表。
def createCk(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            # 获取两个集合的前k-2个项
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            # 如果前k-2个项相同，则将两个集合合并
			# 将合并结果添加到返回列表中
			# 添加使用append函数，合并集合使用
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])
    
    return retList

# apriori函数
def apriori(dataSet, minSupport=0.5):
    # 生成单个物品项集
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    # 根据单个物品项集获取1-频繁项集及其支持度
    L1,supportData = scanD(D,C1,minSupport)
    # 使用L保存所有的频繁项集
    L = [L1]
    # k为项集中每一项物品数
    k = 2
    while len(L[k-2]) > 0:
        # 生成k-候选集
        Ck = createCk(L[k-2],k)
        # 调用scanD函数获取k-频繁项集及其支持度
        # 保存到变量Lk和supK中
        Lk,supK = scanD(D,Ck,minSupport)
        # 更新支持度字典supportData、频繁项集列表L以及k
        # 支持度字典使用update()函数，频繁项集使用append()函数
        supportData.update(supK)
        L.append(Lk)
        k += 1
    
    return L,supportData

# 主函数
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1,len(L)):
        # 注意这里的i是(i+1)-频繁项集的下标
        for freqSet in L[i]:
            # 构造只包含单个元素集合的列表
            H1 = [frozenset([item]) for item in freqSet]
            if (i>1):
                mergeRules(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

# 置信度计算及修剪函数
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 创建修剪后的规则列表H
    prundeH = []
    for conseq in H:
        # 计算以conseq作为右件的规则的置信度conf
        # 使用supportData中的数据即可
        # 提示：conseq作为右件时，左件为freqSet-conseq
        # 左右件同时出现的支持度刚好等于freqSet的支持度
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf>=minConf:
            # 打印规则
            print (freqSet-conseq,'--->',conseq,'conf:',conf)
            # 添加规则
            # 使用append函数，添加内容为（左件，右件，置信度）
            brl.append((freqSet-conseq,conseq,conf))
            prundeH.append(conseq)
            
    return prundeH

# 合并规则函数。函数没有返回值。
def mergeRules(freqSet, H, supportData, brl, minConf):
    # 获取目前规则列表中规则的右件的元素个数
    m = len(H[0])
    # 确保该频繁项集大到可以移除大小为m的子集
    if len(freqSet) > (m+1):
        # 调用createCk函数来生成H中元素的无重复组合
        Hmp1 = createCk(H,m+1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,brl,minConf)
        if len(Hmp1)>1:
            mergeRules(freqSet,Hmp1,supportData,brl,minConf)
            
# 测试函数
def test(minSupport=0.5,minConf=0.7):
    data = loadDataSet()
    L,supportData = apriori(data,minSupport)
    rules = generateRules(L,supportData,minConf)
    return rules

if __name__=="__main__":
    rules = test()