# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import division
from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt


Xi=np.array([6.19, 2.51, 7.29, 7.01, 5.7, 2.66, 3.98, 2.5, 9.1, 4.2])
Yi=np.array([5.25, 2.83, 6.41, 6.71, 5.1, 4.23, 5.05, 1.98, 10.5, 6.3])


def func(p,x,y):
    return p[0]*x+p[1]-y

p0=[1,20]
result = leastsq(func, p0, args=(Xi, Yi))

k,b=result[0]

print(k,b)


plt.figure(figsize=(8,6))
plt.scatter(Xi,Yi,color="blue",label="Data Points",linewidth=2)
x=np.linspace(0,15,100)
y=k*x+b
plt.plot(x,y,color="red",label="Fitting Result",linewidth=2)
plt.legend(loc='lower right')
plt.show()


def leastsq1(x,y):
    """
    x,y分别是要拟合的数据的自变量列表和因变量列表
    """
    meanx = sum(x)/ len(x)   #求x的平均值
    meany = sum(y) / len(y)   #求y的平均值

    xsum = 0.0
    ysum = 0.0

    for i in range(len(x)):
        xsum += (x[i] - meanx)*(y[i]-meany)
        ysum += (x[i] - meanx)**2

    k = xsum/ysum
    b = meany - k*meanx

    return k,b   #返回拟合的两个参数值

k,b = leastsq1(Xi,Yi)

print(k,b)

plt.figure(figsize=(8,6))
plt.scatter(Xi,Yi,color="blue",label="Data Points",linewidth=2)
x=np.linspace(0,15,100)
y=k*x+b
plt.plot(x,y,color="yellow",label="Fitting Result",linewidth=2)
plt.legend(loc='lower right')
plt.show()



