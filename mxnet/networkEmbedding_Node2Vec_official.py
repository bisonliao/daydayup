#############################################################
# 官方包node2vec的实现
# coding:utf8

import csv
from  scipy.sparse import csr_matrix,lil_matrix
import fasttext
import random
from networkx import from_scipy_sparse_matrix, draw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from  sklearn.cluster import DBSCAN
from  sklearn.cluster import KMeans
import networkx as nx
from node2vec import Node2Vec
from gensim.models.word2vec import Word2Vec

DATA_DIR= "E:\\DeepLearning\\data\\network_data\\BlogCatalog-dataset\\data\\"
DIM=20
TEST_USER=2
P = 1
Q = 1
TRAIN=False

# 把csv数据加载为二维array，是一个稀疏矩阵
def loadData():
    maxUserId = int(0)
    with open(DATA_DIR+"nodes.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) > maxUserId:
                maxUserId = int(row[0])
    print("maxUserId:", maxUserId)
    #maxUserId = 14 #小网络直观验证开启
    edges = lil_matrix((maxUserId+1, maxUserId+1), dtype="uint8")
    with open(DATA_DIR+"edges.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) > maxUserId or int(row[1]) > maxUserId:
                continue
            edges[ int(row[0]), int(row[1])] = 1
            edges[int(row[1]), int(row[0])] = 1
    return edges
def jaccardSimilarity(A:set,B:set):
    return len(A.intersection(B)) / (len(A.union(B))+0.001)
def cosDist(v1:np.ndarray,v2:np.ndarray):
    return 1 - np.dot(v1,v2)/(np.linalg.norm(v1,ord=2)*np.linalg.norm(v2, ord=2))
#检查聚类效果，高内聚、低耦合
def clusterEffection(Y, U, cluster):

    # 检查簇1 簇2，看看簇内的平均距离，和簇间的平均距离
    c1 = list()
    c2 = list()
    for i in range(1, U.shape[0]):
        if cluster[i-1] == 5:
            c1.append(i)
        elif cluster[i-1] == 2:
            c2.append(i)
    if len(c1) < 3 or len(c2) < 3:
        print("cluster too small!")
        return
    print("c1, c2 size:%d,%d"%(len(c1),len(c2)))
    sum = 0
    cnt = 0
    for i in range(len(c1)-1):
        for j in range(i+1, len(c1)):
            a = c1[i]
            b = c1[j]
            a = U[a]
            b = U[b]
            sum += cosDist(a, b)
            cnt += 1
    print("%d avg cos distances in cluster:%.2f"%(cnt, sum / cnt))
    sum = 0
    cnt = 0
    for i in range(len(c1)):
        for j in range(len(c2)):
            a = c1[i]
            b = c2[j]
            a = U[a]
            b = U[b]
            sum += cosDist(a, b)
            cnt += 1
    print("%d avg cos distances between cluster:%.2f" % (cnt, sum / cnt))
    ###########################
    # 检查簇内边的密度和簇间边的密度
    edgeNum = 0
    cnt = 0
    for i in range(len(c1) - 1):
        for j in range(i + 1, len(c1)):
            a = c1[i]
            b = c1[j]
            if Y[a, b] > 0:
                edgeNum += 1
            cnt += 1
    print("%d avg edges dense in cluster:%.5f" % (cnt, edgeNum / cnt))
    edgeNum = 0
    cnt = 0
    for i in range(len(c1)):
        for j in range(len(c2)):
            a = c1[i]
            b = c2[j]
            if Y[a, b] > 0:
                edgeNum += 1
            cnt += 1
    print("%d avg edges dense between cluster:%.5f" % (cnt, edgeNum / cnt))

edges = loadData()  # type:lil_matrix
print("edges number:", edges.nnz)
if TRAIN:
    #下面两行都可以正确初始化一个graph对象
    #g = nx.read_adjlist(DATA_DIR+"edges2.csv", delimiter=",", nodetype=int)# type:nx.classes.graph.Graph
    g=from_scipy_sparse_matrix(edges)
    n2v = Node2Vec(g, dimensions=DIM, walk_length=20, num_walks=30, p=P, q=Q)
    model2 = n2v.fit(window=6, min_count=1, batch_words=4) #这里居然是单核训练的，很低效。前面fasttext是多核的

    print(type(model2))
    model2.save("./data/node2vec.bin")


model2 = Word2Vec.load("./data/node2vec.bin")
print(model2.wv.most_similar("2", topn=10))
m = np.zeros((edges.shape[0], DIM))
for i in range(1, edges.shape[0]):
    w = str(i)
    m[i] = model2.wv.get_vector(w)
#cl = DBSCAN(metric='cosine', eps=0.25)
cl = KMeans(n_clusters=100)
cluster = cl.fit_predict(m)
print(cluster)
clusterEffection(edges, m, cluster)

if DIM == 2 and edges.shape[0] < 100:
    draw(g, pos=m, with_labels=True)
    plt.show()