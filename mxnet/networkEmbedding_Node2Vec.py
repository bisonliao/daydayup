'''
对网络实现Node2Vec算法
用到的数据来自：http://socialcomputing.asu.edu/pages/datasets
算法第二步：n-gram 算法进行词向量学习，使用的是fasttext

用一个小的网络验证，通过图形展示节点位置、聚类的结果可以看出该算法和代码是有效的。
fasttext训练过程中，lr等几个参数特别关键，需要注意。lr与epoch有关
'''
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

def getNextNode(edges:lil_matrix, v, t, xs:np.ndarray) :
    if P==1 and Q==1 or t == -1:
        index = random.randint(0, len(xs) - 1)
        return index

    ratio = np.zeros_like(xs, dtype="float32")
    for i in range(len(xs)):
        if xs[i] == t:
            ratio[i] = 1/P
        elif edges[xs[i], t] > 0:
            ratio[i] = 1
        else:
            ratio[i] = 1/Q
    r = random.randint(0, 1000000) / 1000000 * ratio.sum()
    for index in range(len(xs)):
        if r <= 0:
            return index
        r -= ratio[index]
    return index



# 输入稀疏矩阵，输出一个词库文件：把network embedding问题转化为词向量问题
# 这段代码会比较占用时间，尝试改为多线程，但试验后发现没有加快，受限于python解释器的机制
def geneCorpusFromEdges(edges:lil_matrix):
    maxUserId = edges.shape[0]-1
    WORD_NUM = 20
    EXAMP_NUM = maxUserId * 30
    example_list = list()
    for i in range(EXAMP_NUM):
        r = random.randint(1, maxUserId)
        user = r

        wordList = list()
        wordList.append("u%d"%user)
        for j in range(WORD_NUM):
            row = edges[user]
            indices = row.nonzero()[1]
            if len(indices) < 1:
                break
            if len(wordList) > 1 :
                t = int(wordList[-2][1:])
            else:
                t = -1

            index = getNextNode(edges, user, t, indices)
            user = indices[index]
            wordList.append("u%d" % user)
        example_list.append(" ".join(wordList))
        print("\rgenerate example process %.2f%%"%(i*100/EXAMP_NUM), end='')
    print("\n")

    with open("./data/ne_corpus.txt", "w", encoding='utf8') as f:
        [f.write(ex+"\n") for ex in example_list ]


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

def test(model, edges):
    w = "u"+str(TEST_USER)
    A = set(edges[TEST_USER].nonzero()[1])
    nn = list(model.get_nearest_neighbors(w, k=5))
    print(nn)

    cluster(model, edges)

    if DIM == 2 and edges.shape[0] < 100:
        graph = from_scipy_sparse_matrix(edges)
        embedding = np.zeros((edges.shape[0], DIM) )
        for i in range(edges.shape[0]-1):
            embedding[i+1] = model.get_word_vector("u"+str(i+1))
        draw(graph, pos=embedding, with_labels=True)
        plt.show()

def cluster(model:fasttext.FastText._FastText,edges):
    m = np.zeros((edges.shape[0], DIM))
    for i in range(1, edges.shape[0]):
        w = "u"+str(i)
        m[i] = model.get_word_vector(w)
    #cl = DBSCAN( metric='cosine', eps=0.15)
    cl = KMeans(n_clusters=100)
    cluster = cl.fit_predict(m)
    print(cluster)
    clusterEffection(edges, m, cluster)


edges = loadData()  # type:lil_matrix
print("edges number:", edges.nnz)
if TRAIN:
    #geneCorpusFromEdges(edges)
    #print("geneCorpusFromEdges() done!")
    #model = fasttext.train_unsupervised("./data/ne_corpus.txt", epoch = 200, lr=0.0001, dim=DIM,maxn=0) # type:fasttext.FastText._FastText
    #model = fasttext.train_unsupervised("./data/ne_corpus.txt", epoch=100000, lr=0.0001, dim=DIM, maxn=0)  # type:fasttext.FastText._FastText
    model = fasttext.train_unsupervised("./data/ne_corpus.txt", epoch=400, lr=0.1, dim=DIM, maxn=0)  # type:fasttext.FastText._FastText
    model.save_model("./data/network_embedding.bin")
    print(type(model))

model = fasttext.load_model("./data/network_embedding.bin")
A = test(model, edges)










