'''
LINE-1st算法
'''
# coding: utf-8
import numpy as np
import csv
from  scipy.sparse import csr_matrix,lil_matrix
import mxnet.ndarray as nd
import mxnet as mx
import mxnet.autograd as autograd
import pickle
import random
import time
import datetime


DATA_DIR= "E:\\DeepLearning\\data\\network_data\\BlogCatalog-dataset\\data\\"
MATRIX_SZ = 10313
#MATRIX_SZ=15
DIM = 20
CLUSTER_NUM = 100
context = mx.gpu(0)
lr = 0.005
epochs=10000


# 把csv数据加载为二维array，是一个稀疏矩阵
def loadData():
    maxUserId = int(0)
    with open(DATA_DIR + "nodes.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) > maxUserId:
                maxUserId = int(row[0])
    print("maxUserId:", maxUserId)
    maxUserId = MATRIX_SZ-1 #小网络直观验证开启
    edges = lil_matrix((maxUserId+1, maxUserId+1), dtype="uint8")
    with open(DATA_DIR+"edges.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) > maxUserId or int(row[1]) > maxUserId:
                continue
            edges[ int(row[0]), int(row[1])] = 1
            edges[int(row[1]), int(row[0])] = 1
    return edges
############################################################
# 当矩阵比较小，不需要分块计算的时候，U和V实际上是同一个矩阵变量
# 当矩阵比较大，分块计算的时候，U和V是同一个矩阵的不同区域
def my_loss(adj:nd.NDArray, U:nd.NDArray, V:nd.NDArray):

    # 有边相连的两个顶点向量的点积要尽量的大，也就是cos相似度尽量的大
    diff = nd.dot(U,V.transpose())*(-1) #type:nd.NDArray
    diff = 1/(1+diff.exp())#type:nd.NDArray
    diff = diff.log()
    diff = adj * diff * (-1)

    # LINE-1st是没有这一项的，发现聚类效果不好。我修正一下：
    # 没有边相连的两个顶点向量的点积的绝对值要尽量的小，也就是cos相似度为0
    diff2 = (nd.ones_like(adj, ctx=context) - adj)*nd.dot(U,V.transpose())
    return  diff.sum()+nd.norm(diff2, ord=2)


def train(adjmatrix:nd.NDArray, epochs):
    U = nd.random.uniform(0, 1, shape=(adjmatrix.shape[0], DIM), ctx=context)
    SZ = adjmatrix.shape[0]*DIM
    lossum = 0
    for e in range(epochs):
        U.attach_grad()
        with autograd.record():
            L = my_loss(adjmatrix, U, U)
        L.backward()
        lossum += L.asscalar()
        GAP = 1000
        if e == 0:
            print("ep:%d, loss:%.10f"%(e,lossum/SZ))
        if e % GAP == 0 and e != 0 :
            avgLoss = lossum/GAP/SZ
            lossum = 0
            print("ep:%d, loss:%.10f"%(e,avgLoss))
        U = U - lr * U.grad
    return U

#与上面train()的区别是：为了解决内存放不下的问题，分小块进行梯度下降。
def train_partition(adjmatrix, epochs):
    U = nd.random.uniform(0, 1, shape=(adjmatrix.shape[0], DIM), ctx=mx.cpu(0))
    matSize = adjmatrix.shape[0]
    PART_SZ = 6000 #分块的大小， 根据显卡内存大小做适当调整
    part_num = matSize // PART_SZ
    if matSize % PART_SZ > 0:
        part_num += 1
    for e in range(epochs):
        lossum = 0
        for i in range(part_num):
            x = i*PART_SZ
            width = PART_SZ
            if x+width > matSize:
                width = matSize-x
            left = U[x:x+width].as_in_context(context)
            for j in range(part_num):
                y = j * PART_SZ
                height = PART_SZ
                if y+height > matSize:
                    height = matSize-y
                right = U[y:y+height].as_in_context(context)

                left.attach_grad()
                right.attach_grad()
                with autograd.record():
                    L = my_loss(adjmatrix[x:x+width,y:y+height], left, right)
                L.backward()
                lossum += L.asscalar()
                left = left - lr * left.grad
                U[x:x + width] = left.as_in_context(mx.cpu(0))
                if i != j:
                    right = right - lr*right.grad
                    U[y:y + height] = right.as_in_context(mx.cpu(0))

        if e % 100 == 0:
            print(datetime.datetime.now().strftime('%H:%M:%S'), " ep:%d, loss:%.10f"%(e,lossum/matSize/matSize))
    return U.as_in_context(context)

if False:
    m = loadData() # type:lil_matrix
    Y = nd.array(m.toarray(),ctx=context)

    if Y.shape[0] > 1000:
        U = train_partition(Y, epochs)#type:nd.NDArray
    else:
        U = train(Y, epochs) #type:nd.NDArray

    with open("./data/LINE_U.data", "wb") as f:
        pickle.dump(U, f)
    with open("./data/LINE_Y.data", "wb") as f:
        pickle.dump(Y, f)
else:
    with open("./data/LINE_U.data", "rb") as f:
        U = pickle.load(f)
    with open("./data/LINE_Y.data", "rb") as f:
        Y = pickle.load(f)

###########################################################
## 训练完了，就要用聚类、查询相似节点、可视化等各种手段检查embedding是否符合预期了
from  sklearn.cluster import KMeans
from  sklearn.cluster import DBSCAN
from  sklearn.neighbors import BallTree
from networkx import from_scipy_sparse_matrix, draw, from_numpy_array
import matplotlib.pyplot as plt


def cosDist(v1:np.ndarray,v2:np.ndarray):
    return 1 - np.dot(v1,v2)/(np.linalg.norm(v1,ord=2)*np.linalg.norm(v2, ord=2))

#检查聚类效果，高内聚、低耦合
def clusterEffection(Y, U, cluster):

    cls = list()
    for i in range(CLUSTER_NUM):
        cls.append(list())
    for i in range(len(cluster)):
        cls[ cluster[i] ].append(i+1)
    random.shuffle(cls)

    #簇内边密度均值
    edgeNum = 0
    cnt = 0
    for k in range(CLUSTER_NUM):
        c1 = cls[k]
        if len(c1) < 3 or len(c1)>100: #为了计算高效，只抽查大小适中的簇
            continue

        for i in range(len(c1) - 1):
            for j in range(i + 1, len(c1)):
                a = c1[i]
                b = c1[j]
                if Y[a, b] > 0:
                    edgeNum += 1
                cnt += 1
    print("avg edges dense in cluster:%.5f" % (edgeNum / (0.001+cnt)))

    #簇间边密度均值
    edgeNum = 0
    cnt = 0
    for k in range(CLUSTER_NUM):
        c1 = cls[k]
        if len(c1) < 3  or len(c1)>100:
            continue
        for z in range(k+1, CLUSTER_NUM):
            c1 = cls[k]
            c2 = cls[z]
            if len(c2) < 3  or len(c2)>100: #为了计算高效，只抽查大小适中的簇
                continue

            for i in range(len(c1)):
                for j in range(len(c2)):
                    a = c1[i]
                    b = c2[j]
                    if Y[a, b] > 0:
                        edgeNum += 1
                    cnt += 1
            if cnt > 100000:
                break
        if cnt > 100000:
            break
    print("avg edges dense between cluster:%.5f" % (edgeNum / (cnt+0.001)))


######聚类的结果
U = U.asnumpy()
cl = KMeans(n_clusters=CLUSTER_NUM)
# 通过调整eps，使得这1万多个点聚类为17个簇，同时有1314个点被认为是噪声点
#cl = DBSCAN( metric='cosine', eps=0.28)
cluster = cl.fit_predict(U[1:])
print("cluster labels:", cluster)
print("noisy node:", np.array([1 if c == -1 else 0 for c in cluster]).sum()) #有多少个被聚类算法认为是噪声？
clusterEffection(Y, U, cluster)

### 可视化 与 相似节点的查询结果
if DIM == 2 and Y.shape[0] < 100:
    tree = BallTree(U, metric='pyfunc', func=cosDist)
    print(tree.query([U[2]], k=7))
    graph = from_numpy_array(Y.asnumpy())
    draw(graph, pos=U, with_labels=True)
    plt.show()


