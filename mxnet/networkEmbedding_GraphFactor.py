'''
将邻接矩阵 Y 分解为 U 点乘 U的转置的形式
U的每一行就是一个节点的embedding
'''
import numpy as np
import csv
from  scipy.sparse import csr_matrix,lil_matrix
import mxnet.ndarray as nd
import mxnet as mx
import mxnet.autograd as autograd

DATA_DIR= "E:\\DeepLearning\\data\\network_data\\BlogCatalog-dataset\\data\\"
MATRIX_SZ = 10313
#MATRIX_SZ=15
DIM = 20
context = mx.gpu(0)
lr = 0.00005
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
    return diff    +nd.sum(nameda * U*U)  #后项是L2正则项防止过拟合


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

def createFlag(adjMatrix):
    flags = np.zeros((MATRIX_SZ,MATRIX_SZ))
    for i in range(MATRIX_SZ):
        cnt = 0
        for j in range(MATRIX_SZ):
            if cnt < DIM and adjMatrix[i,j] > 0:
                flags[i,j] = 1
                cnt += 1
        for j in range(MATRIX_SZ):
            if cnt < DIM and adjMatrix[i,j] == 0:
                flags[i,j] = 1
                cnt += 1

    return nd.array(flags, ctx=context)


m = loadData() # type:lil_matrix
for i in range(MATRIX_SZ):
    m[i,i] = 1
Y = nd.array(m.toarray(),ctx=context)
if MATRIX_SZ > 1000: #太大了，分块训练
    U = nd.zeros(shape=(Y.shape[0], DIM), ctx=context)
    for i in range(MATRIX_SZ//1000):
        start = i*1000
        end = start+1000
        if end > MATRIX_SZ:
            end = MATRIX_SZ
        flags = nd.ones((end-start, end-start), ctx=context)
        U[start:end] = train(Y[start:end,start:end],epochs, flags)

else:
    flags = nd.ones((MATRIX_SZ, MATRIX_SZ), ctx=context)
    U = train(Y, epochs, flags) #type:nd.NDArray

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
######聚类的结果
U = U.asnumpy()
if Y.shape[0] < 100:
    cl = KMeans(n_clusters=3)
    print(cl.fit_predict(U[1:]))

##### 相似节点的查询结果
tree = BallTree(U, metric='pyfunc', func=cosDist)
print(tree.query([U[2]],k=7))

### 可视化
if DIM == 2 and Y.shape[0] < 100:
    graph = from_numpy_array(Y.asnumpy())
    draw(graph, pos=U, with_labels=True)
    plt.show()


