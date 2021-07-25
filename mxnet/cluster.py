# -*- coding:utf-8 -*-

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import dbscan

m = 100
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

core_samples, cluster_ids = dbscan(X_moons, eps = 0.4, min_samples=10)
print(cluster_ids)

def draw():
    global X_moons,cluster_ids
    xx = []
    yy = []
    i = 0
    for sample in X_moons:
        x1, x2 = sample
        y = cluster_ids[i]
        i += 1
        if y == 0:
            xx.append(x1)
            yy.append(x2)
    plt.scatter(xx, yy, color="r")

    xx = []
    yy = []
    i = 0
    for sample in X_moons:
        x1, x2 = sample
        y = cluster_ids[i]
        i += 1
        if y == 1:
            xx.append(x1)
            yy.append(x2)
    plt.scatter(xx, yy, color="b")

    xx = []
    yy = []
    i = 0
    for sample in X_moons:
        x1, x2 = sample
        y = cluster_ids[i]
        i += 1
        if y ==-1:
            xx.append(x1)
            yy.append(x2)
    plt.scatter(xx, yy, color="g")

    plt.show()

draw()