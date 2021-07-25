# -*- coding:utf-8 -*-
# 手工实现逻辑斯蒂回归

from sklearn.datasets import make_moons
import math
import numpy as np
import matplotlib.pyplot as plt


#from sklearn.linear_model.logistic import  LogisticRegression




m = 100
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
data = [[ X_moons[i][0], X_moons[i][1], y_moons[i]] for i in range(m)]

for i in range(4):
    print(data[i])


'''classifer = LogisticRegression()
classifer.fit(X_moons,y_moons)
predictions = classifer.predict(X_moons)
cnt = 0
for p in predictions == y_moons:
    if p :
        cnt += 1
print(cnt/m)
print(classifer)'''

lr = 10
th2 = 0.3
th1 = 0.5
th0 = 0.1


def h(x1, x2):
    global m, data, lr, th0, th1,th2
    v = th0 + th1 * x1 + th2 * x2
    return 1/(1+math.pow(math.e, -v))

def cost():
    global m, data, lr, th0, th1, th2
    c = 0
    for sample in data:
        x1, x2, y = sample
        c = c + ( - y * math.log(h(x1, x2)) - (1-y)*math.log(1-h(x1, x2)) )
    return c / m

def deriv0():
    global m, data, lr, th0, th1, th2
    d = 0
    for sample in data:
        x1, x2, y = sample
        d += (h(x1,x2)-y)*1
    return d/m


def deriv1():
    global m, data, lr, th0, th1, th2
    d = 0
    for sample in data:
        x1, x2, y = sample
        d += (h(x1, x2) - y) * x1
    return d / m

def deriv2():
    global m, data, lr, th0, th1, th2
    d = 0
    for sample in data:
        x1, x2, y = sample
        d += (h(x1, x2) - y) * x2
    return d / m

def draw():
    global m, data, lr, th0, th1, th2
    xx = []
    yy = []
    for sample in data:
        x1, x2, y = sample
        if y == 0:
            continue
        xx.append(x1)
        yy.append(x2)
    plt.scatter(xx, yy, color="r")

    xx = []
    yy = []
    for sample in data:
        x1, x2, y = sample
        if y == 1:
            continue
        xx.append(x1)
        yy.append(x2)
    plt.scatter(xx, yy, color="b")
    plt.show()


def draw2():
    global m, data, lr, th0, th1, th2
    xx = []
    yy = []
    for sample in data:
        x1, x2, _ = sample
        y = h(x1, x2)
        if y < 0.5 :
            continue
        xx.append(x1)
        yy.append(x2)
    plt.scatter(xx, yy, color="r")

    xx = []
    yy = []
    for sample in data:
        x1, x2, _ = sample
        y = h(x1, x2)
        if y >= 0.5:
            continue
        xx.append(x1)
        yy.append(x2)
    plt.scatter(xx, yy, color="b")

    plt.show()


def main():
    global m, data, lr, th0, th1, th2
    for i in range(1, 1000):
        if (i % 97) == 1:
            print(i, " cost:", cost())
        th0 = th0 - lr * deriv0()
        th1 = th1 - lr * deriv1()
        th2 = th2 - lr * deriv2()
    print("th0:", th0, " th1:", th1, " th2:", th2," cost:", cost())
    draw()
    draw2()
main()





