'''
使用梯度下降的方法求解方阵的逆
类似的，还可以求解矩阵的 UV分解
'''
import csv
import random
import pickle
import mxnet.ndarray as nd
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon as gluon
import math

size = 20
epoch = 100000
context = mx.gpu(0)
lr = 0.1
A = nd.random.normal(10, 3, shape=(size, size), ctx=context)

def getUnitMatrix(sz):
    I = nd.zeros(shape=(sz, sz), ctx = context)
    for idx in range(sz):
        I[idx, idx] = 1
    return I

unitMatrix = getUnitMatrix(size)

B = nd.random.uniform(0, 1, shape=(size, size),ctx=context)
loss = gluon.loss.L2Loss()

for e in range(epoch):
    B.attach_grad()
    with autograd.record():
        Y = nd.dot(A,B)
        L = loss(unitMatrix, Y)
    L.backward()
    if e % 1000 == 0 and e > 0:
        print("ep:", e, " loss:", nd.mean(L).asscalar())
    B = B - (lr/size/size) * B.grad

print(B)
print(nd.dot(A,B))





