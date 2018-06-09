#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 21:09:25 2018

@author: Administrator
"""

import math
import random

random.seed(0)

def rand(a,b):
    return (b-a)*random.random()+a

def make_matrix(m,n,fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill]*n)
    
    return mat

def sigmoid(x):
    return 1.0/(1+math.exp(-x))

def sigmoid_deriviate(x):
    return x*(1-x)

class BPNN:
    # 初始化变量
    def __init__(self):
        # 下面三个变量保存各层神经元个数
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        # 下面三个变量保存各层神经元的输出值
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        # 下面两个变量保存权重
        self.input_weights = []
        self.output_weights = []
        
    # 初始化神经网络
    def setup(self,ni,nh,no):
        # ni,nh,no分别代表输入层、隐含层、输出层
        # 的神经元个数
        # 注意这里只对隐含层的神经元增加偏置值
        # 处理方式是在输入层增加一个神经元，使
        # 其统一到权重之中。
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # 初始化每一层神经元的值为1
        # 之所以设为1是为了将偏置值看作是输入恒为1的
        # 神经元与其它神经元的连接的权重
        self.input_cells = [1.0]*self.input_n
        self.hidden_cells = [1.0]*self.hidden_n
        self.output_cells = [1.0]*self.output_n
        # 初始化权重
        self.input_weights = \
                    make_matrix(self.input_n,self.hidden_n)
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = \
                                rand(-2.0,2.0)
        
        self.output_weights = \
                    make_matrix(self.hidden_n,self.output_n)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = \
                                rand(-2.0,2.0)
                            
    # 编写predict函数进行一次前馈，返回输出
    def predict(self,inputs):
        # 根据用户提供的输入填充输入层的数据
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # 根据输入层的数据前向计算隐含层的值
        for h in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i]*self.input_weights[i][h]
            
            # 调用激活函数
            self.hidden_cells[h] = sigmoid(total)
         
        # 根据隐含层的数据前向计算输出层的值
        for o in range(self.output_n):
            total = 0.0
            for h in range(self.hidden_n):
                total += self.hidden_cells[h]*self.output_weights[h][o]
            
            self.output_cells[o] = sigmoid(total)
            
        return self.output_cells[:]
    
    #执行一次反向传播和权值更新，并返回预测的误差
    def back_propagate(self,case,label,learning_rate):
        # case和label分别是训练数据及期望输出
        # learning_rate为学习率
        # 前向传播填充各层数据
        self.predict(case)
        # 获取输出层误差
        output_error = [0.0]*self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_error[o] = error*\
                    sigmoid_deriviate(self.output_cells[o])
        
        # 获取隐含层误差
        hidden_error = [0.0]*self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_error[o] * self.output_weights[h][o]
                hidden_error[h] = error*\
                        sigmoid_deriviate(self.hidden_cells[h])
        # 更新隐含层到输出层权值
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_error[o]*self.hidden_cells[h]
                self.output_weights[h][o]+=\
                        learning_rate * change
        
        # 更新输入层到隐含层权值
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_error[h]*self.input_cells[i]
                self.input_weights[i][h]+=\
                        learning_rate * change
        
        # 返回全局误差
        global_error = 0.0
        for o in range(self.output_n):
            global_error += 0.5*(label[o]-\
                                 self.output_cells[o])**2
        
        return global_error
    
    # 定义训练函数
    def train(self,cases,labels,limit=10000,lr=0.1):
        #  limit代表迭代次数，lr是学习率
        for i in range(limit):
            error = 0.0
            for j in range(len(cases)):
                label = labels[j]
                case = cases[j]
                # 调用反向传播函数，累加误差
                # 当迭代次数是1000的倍数时打印误差
                error += self.back_propagate(case,label,lr)
                if i%1000==0:
                    #pass
                    print(error)
        
    # 定义测试函数，目的是学习异或逻辑
    def test(self):
        cases = [[0,0],
                 [0,1],
                 [1,0],
                 [1,1]]
        labels = [[0],[1],[1],[0]]
        
        self.setup(2,5,1)
        self.train(cases,labels)
        for case in cases:
            print(self.predict(case))
    
if __name__=="__main__":
    nn = BPNN()
    nn.test()
        
        
