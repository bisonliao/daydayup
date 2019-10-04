'''
梯度下降的方法没有搞定, U退化为0，加约束的正则项也不怎么起作用
用特征值特征向量的方法可以搞定
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


DATA_DIR= "E:\\DeepLearning\\data\\network_data\\BlogCatalog-dataset\\data\\"
MATRIX_SZ = 10313
#MATRIX_SZ=15
DIM = 20
CLUSTER_NUM = 100
context = mx.gpu(0)
lr = 0.005
epochs=20000

# 把csv数据加载为二维array，是一个稀疏矩阵
def loadData():
    maxUserId = int(0)
    with open(DATA_DIR + "nodes.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) > maxUserId:
                maxUserId = int(row[0])
    print("maxUserId:", maxUserId)
    #maxUserId = MATRIX_SZ-1 #小网络直观验证开启
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

def my_loss(adj, U):
    diff = U - nd.dot(adj / adj.shape[0], U)
    diff2 = 1 / ( U + nd.ones_like(U)*0.00001)#这一项防止U趋近0

    return  nd.norm(diff, 2)+nd.norm(diff2, 2)


def train(adjmatrix, epochs):
    U = nd.random.uniform(1, 10, shape=(adjmatrix.shape[0], DIM), ctx=context)
    SZ = adjmatrix.shape[0] * DIM
    lossum = 0
    for e in range(epochs):
        U.attach_grad()
        with autograd.record():
            L = my_loss(adjmatrix,U)
        L.backward()
        lossum += L.asscalar()
        GAP = 1000
        if e == 0:
            print("ep:%d, loss:%.4f"%(e,lossum/SZ))
        if e % GAP == 0 and e != 0 :
            avgLoss = lossum/GAP/SZ
            lossum = 0
            print("ep:%d, loss:%.4f"%(e,avgLoss))
        U = U - lr * U.grad
    return U



if False:
    m = loadData() # type:lil_matrix
    Y = nd.array(m.toarray(),ctx=context)

    w,vec = np.linalg.eig(m.toarray())


    '''if MATRIX_SZ > 1000: #太大了，分块训练
        U = train_partition(Y, epochs)
    else:
        U = train(Y, epochs) #type:nd.NDArray
    '''
    U = nd.array(vec[:, :DIM], ctx=context)
    with open("./data/localLinear_U.data", "wb") as f:
        pickle.dump(U, f)
    with open("./data/localLinear_Y.data", "wb") as f:
        pickle.dump(Y, f)
else:
    with open("./data/localLinear_U.data", "rb") as f:
        U = pickle.load(f)
    with open("./data/localLinear_Y.data", "rb") as f:
        Y = pickle.load(f)

###########################################################
## 训练完了，就要用聚类、查询相似节点、可视化等各种手段检查embedding是否符合预期了
if DIM == 2 and Y.shape[0] < 100:
    print(U)
    print(nd.dot(Y,U))
    print( (U - nd.dot(Y,U))/(U+0.001))

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
    tree = BallTree(U[1:], metric='pyfunc', func=cosDist)
    print(tree.query([U[2]], k=7))
    graph = from_numpy_array(Y.asnumpy())
    draw(graph, pos=U, with_labels=True)
    plt.show()


