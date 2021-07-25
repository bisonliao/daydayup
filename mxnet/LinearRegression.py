# -*- coding:utf-8 -*-
#手工实现线性回归
import numpy as np
import matplotlib.pyplot as plt

data = [(1, 1), (2, 2), (3, 4), (4, 3), (5, 5.5), (6, 8), (7, 6), (8, 8.4), (9, 10), (5, 4)]

th1 = 0.67
th0 = -1
lr = 0.001




def h(x):
    global data, th1, th0, lr
    return th1 * x + th0


def cost():
    global data, th1, th0, lr
    c = 0
    for sample in data:
        x, y = sample
        c += (y - h(x)) * (y - h(x))
    return c / 2 / len(data)

# 对涩塔0 的导数
def deriv0():
    global data, th1, th0, lr
    d = 0
    for sample in data:
        x, y = sample
        d += h(x) - y
    return d / len(data)

# 对涩塔1 的导数
def deriv1():
    global data, th1, th0, lr
    d = 0
    for sample in data:
        x, y = sample
        d += (h(x) - y) * x
    return d / len(data)

def draw():
    global data, th1, th0, lr
    xx = []
    yy = []
    for sample in data:
        x, y = sample
        xx.append(x)
        yy.append(y)
    plt.scatter(xx, yy)

    xx = []
    yy = []
    for x in [0, 10]:
        y = th0 + th1 * x
        xx.append(x)
        yy.append(y)
    plt.plot(xx, yy, color="r")
    plt.show()



def main():
    global data, th1, th0, lr
    for i in range(1, 10000):
        if (i % 99) == 1:
            print(i, " cost:", cost())
        th0 = th0 - lr * deriv0()
        th1 = th1 - lr * deriv1()
    print("th0:", th0, " th1:", th1, " cost:", cost())
    draw()

main()
