'''
没有搞定, U退化为0，加约束的正则项也不怎么起作用
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
#MATRIX_SZ = 10313
MATRIX_SZ=15
DIM = 2
CLUSTER_NUM = 3
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
    maxUserId = MATRIX_SZ-1 #小网络直观验证开启
    edges = lil_matrix((maxUserId+1, maxUserId+1), dtype="uint8")
    with open(DATA_DIR+"edges2.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) > maxUserId or int(row[1]) > maxUserId:
                continue
            edges[ int(row[0]), int(row[1])] = 1
            edges[int(row[1]), int(row[0])] = 1
    return edges
############################################################

def my_loss(adj, U):
    diff = U - nd.dot(adj, U)
    diff2 = 0.01 / ( U + nd.ones_like(U)*0.00001)

#这一项防止U趋近0
    '''diff2 = 0.001 / ( U + nd.ones_like(U)*0.00001)
    diff2 = nd.norm(diff2, 2)'''

    return  nd.norm(diff, 2)


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

#与上面train()的区别是：为了解决内存放不下的问题，分小块进行梯度下降。
def train_partition(adjmatrix, epochs):
    U = nd.random.uniform(0, 1, shape=(adjmatrix.shape[0], DIM), ctx=mx.cpu(0))
    matSize = adjmatrix.shape[0]
    PART_SZ = 5000 #分块的大小， 根据显卡内存大小做适当调整
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
                flags = nd.ones((width,height),ctx=context) #有边没边都要逼近
                #flags = adjmatrix[x:x+width,y:y+height].as_in_context(context) #只考虑有边相连

                left.attach_grad()
                right.attach_grad()
                with autograd.record():
                    Y = nd.dot(left, right.transpose())
                    L = my_loss(Y, adjmatrix[x:x+width,y:y+height],flags,  left)
                L.backward()
                lossum += L.asscalar()
                left = left - lr * left.grad
                U[x:x + width] = left.as_in_context(mx.cpu(0))
                if i != j:
                    right = right - lr*right.grad
                    U[y:y + height] = right.as_in_context(mx.cpu(0))

        if e % 100 == 0:
            print("ep:%d, loss:%.4f"%(e,lossum/matSize/matSize))
    return U.as_in_context(context)

#测试上面两个关键函数的代码
def test():
    global DIM
    global lr
    DIM = 20
    lr = 1e-7
    adjmatrix = nd.random_uniform(1,10, (10000,10), context )
    adjmatrix = nd.dot(adjmatrix, adjmatrix.transpose())
    U = train_partition(adjmatrix, 30000)
    diff = (nd.dot(U, U.transpose())-adjmatrix)/adjmatrix
    print(diff.max())
    print(diff.min())
    exit(0)


if True:
    m = loadData() # type:lil_matrix
    Y = nd.array(m.toarray(),ctx=context)
    if MATRIX_SZ > 1000: #太大了，分块训练
        U = train_partition(Y, epochs)
    else:
        U = train(Y, epochs) #type:nd.NDArray
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
    tree = BallTree(U, metric='pyfunc', func=cosDist)
    print(tree.query([U[2]], k=7))
    graph = from_numpy_array(Y.asnumpy())
    draw(graph, pos=U, with_labels=True)
    plt.show()


