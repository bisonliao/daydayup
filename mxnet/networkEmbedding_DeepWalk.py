'''
对网络实现DeepWalk算法
用到的数据来自：http://socialcomputing.asu.edu/pages/datasets
DeepWalk的第二步：n-gram 算法进行词向量学习，使用的是fasttext

同时使用开源的DeepWalk命令行（https://github.com/phanein/deepwalk）另外
生成network embedding做AB对比，发现节点58的两种算法下的最近节点没有相交，我的实现看来有问题。、
另外fasttext训练过程中，loss值没有最终收敛到很小。
'''
# coding:utf8

import csv
from  scipy.sparse import csr_matrix,lil_matrix
import fasttext
import random
from networkx import from_scipy_sparse_matrix, draw
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import BallTree

DATA_DIR= "E:\\DeepLearning\\data\\network_data\\BlogCatalog-dataset\\data\\"
DIM=100

# 把csv数据加载为二维array，是一个稀疏矩阵
def loadData():
    maxUserId = int(0)
    with open(DATA_DIR+"nodes.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) > maxUserId:
                maxUserId = int(row[0])
    print("maxUserId:", maxUserId)
    #maxUserId = 300
    edges = lil_matrix((maxUserId+1, maxUserId+1), dtype="uint8")
    with open(DATA_DIR+"edges.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[0]) > maxUserId or int(row[1]) > maxUserId:
                continue
            edges[ int(row[0]), int(row[1])] = 1
            edges[int(row[1]), int(row[0])] = 1
    return edges


# 输入稀疏矩阵，输出一个词库文件：把network embedding问题转化为词向量问题
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
            index = random.randint(0, len(indices)-1)
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

def test(model, edges):
    w = "u58"
    A = set(edges[int(w[1:])].nonzero()[1])
    nn = list(model.get_nearest_neighbors(w))
    print(nn)
    return [neighbor[1] for neighbor in nn]
    '''for neighbor in nn:
        neighbor = int(neighbor[1][1:])
        B = set(edges[neighbor].nonzero()[1])
        print("%d \tsim:%f"%(neighbor, jaccardSimilarity(A,B)), end='')
        print("\t", B)

    if DIM == 2 and edges.shape[0] < 1000:
        graph = from_scipy_sparse_matrix(edges)
        embedding = np.zeros((edges.shape[0], DIM) )
        for i in range(edges.shape[0]):
            embedding[i] = model.get_word_vector("u"+str(i+1))
        draw(graph, pos=embedding)
        plt.show()'''


edges = loadData()  # type:lil_matrix
print("edges number:", edges.nnz)
if False:
    #geneCorpusFromEdges(edges)
    #print("geneCorpusFromEdges() done!")
    model = fasttext.train_unsupervised("./data/ne_corpus.txt", epoch = 200, lr=0.0001, dim=DIM) # type:fasttext.FastText._FastText
    model.save_model("./data/network_embedding.bin")
    print(type(model))

model = fasttext.load_model("./data/network_embedding.bin")
A = test(model, edges)


# 用开源的命令行工具生成embeding，加载到BallTree里，查询相似节点
def deepwalk(edges):
    embfile = "./data/network.emb" #这个文件是用命令行生成的embedding结果
    f = open(embfile, "r", encoding="utf8")
    cnt = 0
    vects=dict() #读到字典里
    for line in f:
        cnt += 1
        if cnt == 1:
            continue

        if line[-1] =="\n":
            line = line[:-1]
        if line[-1] == "\r":
            line = line[:-1]

        fields = line.split(" ")
        vects[int(fields[0])] = [ float(v) for v in fields[1:] ]
    f.close()
    #把字典转为向量列表
    m = np.zeros((len(vects.keys())+1, DIM), dtype="float32")
    for i in range(len(vects.keys())):
        m[i+1] = np.array(vects[i+1])

    tree = BallTree(m, metric="pyfunc", func=cosDist)
    neighbors = tree.query([m[58]], k=11)
    print(neighbors)
    return neighbors[1][0]

B = deepwalk(edges)
print(jaccardSimilarity(set(A),set(B) ))







