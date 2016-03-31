#! /usr/bin/python

from pandas import DataFrame
import numpy as np
from numpy.random import uniform

CONST_LEFT = -100
CONST_RIGHT = 100

name = input("Dataset name\n")
coefs = np.array([float(i) for i in input("Coefs\n").split()])
N = int(input("N="))

X = []
for i in range(len(coefs)):
    X.append(uniform(low=CONST_LEFT,high=CONST_RIGHT,size=N))
X = np.array(X).T
y = X.dot(coefs)

DataFrame(data=X).to_csv('-'.join(['./datasets/X',name,str(N)])+'.csv',header=False,index=False)
DataFrame(data=y).to_csv('-'.join(['./datasets/Y',name,str(N)])+'.csv',header=False,index=False)
