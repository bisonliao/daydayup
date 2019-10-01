'''
将邻接矩阵 Y 分解为 U 点乘 U的转置的形式
U的每一行就是一个节点的embedding
'''
# coding: utf-8
import numpy as np
import csv
from  scipy.sparse import csr_matrix,lil_matrix
import mxnet.ndarray as nd
import mxnet as mx
import mxnet.autograd as autograd
import pickle

DATA_DIR= "E:\\DeepLearning\\data\\network_data\\BlogCatalog-dataset\\data\\"
MATRIX_SZ = 10313
#MATRIX_SZ=15
DIM = 20
context = mx.gpu(0)
lr = 0.000005
epochs=5000

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
### 使用mxnet做梯度下降的训练，求U满足U点乘U的转置等于 Y
### 有个参数叫flags，是一个与邻接矩阵等形状的矩阵，用于标识邻接矩阵中哪些元素要求逼近（1），哪些不在乎（0）
### 目前的例子里是一个全1的矩阵，即每个元素都要求逼近。
def my_loss(y, label, flags, U):
    nameda = 0.0001
    diff = (y - label)*(y-label)*flags
    diff = nd.sum(diff)
    return diff    #+nd.sum(nameda * U*U)  #后项是L2正则项防止过拟合


def train(adjmatrix, epochs, flags):
    U = nd.random.uniform(0, 1, shape=(adjmatrix.shape[0], DIM), ctx=context)
    lossum = 0
    flagsSum = flags.sum().asscalar()
    print("to learn:", adjmatrix.shape[0]*DIM, " equalization:", flagsSum)
    for e in range(epochs):
        U.attach_grad()

        with autograd.record():
            Y = nd.dot(U,U.transpose())
            L = my_loss(Y, adjmatrix,flags,  U)
        L.backward()
        lossum += L.asscalar()
        GAP = 100
        if e == 0:
            print("ep:%d, loss:%.4f"%(e,lossum/flagsSum))
        if e % GAP == 0 and e != 0 :
            avgLoss = lossum/GAP/flagsSum
            lossum = 0
            print("ep:%d, loss:%.4f"%(e,avgLoss))
        U = U - lr * U.grad
    return U

#与上面train()的区别是：为了解决内存放不下的问题，分小块进行梯度下降。
def train_partition(adjmatrix, epochs):
    U = nd.random.uniform(0, 1, shape=(adjmatrix.shape[0], DIM), ctx=mx.cpu(0))
    PART_SZ = 5000 #分块的大小， 根据显卡内存大小做适当调整
    part_num = MATRIX_SZ // PART_SZ
    if MATRIX_SZ % PART_SZ > 0:
        part_num += 1
    for e in range(epochs):
        lossum = 0
        for i in range(part_num):
            x = i*PART_SZ
            width = PART_SZ
            if x+width > MATRIX_SZ:
                width = MATRIX_SZ-x
            left = U[x:x+width].as_in_context(context)
            for j in range(part_num):
                y = j * PART_SZ
                height = PART_SZ
                if y+height > MATRIX_SZ:
                    height = MATRIX_SZ-y
                right = U[y:y+height].as_in_context(context)
                #flags = nd.ones((width,height),ctx=context) #有边每边都要逼近
                flags = adjmatrix[x:x+width,y:y+height] #只考虑有边相连

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
            print("ep:%d, loss:%.4f"%(e,lossum/MATRIX_SZ/MATRIX_SZ))
    return U.as_in_context(context)




if False:
    m = loadData() # type:lil_matrix
    for i in range(MATRIX_SZ):
        m[i,i] = 1
    Y = nd.array(m.toarray(),ctx=context)
    if MATRIX_SZ > 1000: #太大了，分块训练
        U = train_partition(Y, epochs)
    else:
        flags = nd.ones((MATRIX_SZ, MATRIX_SZ), ctx=context)
        U = train(Y, epochs, flags) #type:nd.NDArray
    with open("./data/U.data", "wb") as f:
        pickle.dump(U, f)
    with open("./data/Y.data", "wb") as f:
        pickle.dump(Y, f)
else:
    with open("./data/U.data", "rb") as f:
        U = pickle.load(f)
    with open("./data/Y.data", "rb") as f:
        Y = pickle.load(f)

###########################################################
## 训练完了，就要用聚类、查询相似节点、可视化等各种手段检查embedding是否符合预期了
if Y.shape[0] < 100:
    v =nd.array(nd.dot(U,U.transpose()), ctx=context)
    for i in range(MATRIX_SZ):
        for j in range(MATRIX_SZ):
            if v[i,j] >= 0.5:
                v[i,j] = 1
            elif v[i,j] <= -0.5:
                v[i, j] = -1
            else:
                v[i,j] = 0
    print((v-Y).as_in_context(context)*flags)




from  sklearn.cluster import KMeans
from  sklearn.cluster import DBSCAN
from  sklearn.neighbors import BallTree
from networkx import from_scipy_sparse_matrix, draw, from_numpy_array
import matplotlib.pyplot as plt

def cosDist(v1:np.ndarray,v2:np.ndarray):
    return 1 - np.dot(v1,v2)/(np.linalg.norm(v1,ord=2)*np.linalg.norm(v2, ord=2))

#检查聚类效果，高内聚、低耦合
def clusterEffection(Y, U, cluster):

    # 检查簇1 簇2，看看簇内的平均距离，和簇间的平均距离
    c1 = list()
    c2 = list()
    for i in range(1, U.shape[0]):
        if cluster[i-1] == 3:
            c1.append(i)
        elif cluster[i-1] == 7:
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



######聚类的结果
U = U.asnumpy()
cl = KMeans(n_clusters=100)
# 通过调整eps，使得这1万多个点聚类为17个簇，同时有1314个点被认为是噪声点
#cl = DBSCAN( metric='cosine', eps=0.28)
cluster = cl.fit_predict(U[1:])
print("cluster labels:", set(cluster))
print("noisy node:", np.array([1 if c == -1 else 0 for c in cluster]).sum()) #有多少个被聚类算法认为是噪声？
clusterEffection(Y, U, cluster)

### 可视化 与 相似节点的查询结果
if DIM == 2 and Y.shape[0] < 100:
    tree = BallTree(U, metric='pyfunc', func=cosDist)
    print(tree.query([U[2]], k=7))
    graph = from_numpy_array(Y.asnumpy())
    draw(graph, pos=U, with_labels=True)
    plt.show()


