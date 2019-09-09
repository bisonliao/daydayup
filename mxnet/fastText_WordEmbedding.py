'''
使用fastText训练词向量
用小说平凡的世界作为语料，使用jieba分词，训练的中文词向量效果不好，可能语料太少
用600M大小的enwiki9.txt作为语料，训练的英文词向量效果简单看还可以，但没有正式的benchmark评测。训练时间比较长，1个多小时。
'''
# coding: UTF-8
import fasttext
import mxnet.gluon as gluon
import jieba
import torch
import numpy as np
import random
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from io import open
import glob
import os
import math
import hashlib
import pickle
import re


def loadData(filename, isChn):
    example_list = list()
    with open(filename, "r", encoding='utf8') as f:
        for line in f:
            line = line[:-1]
            if isChn:
                words = list(jieba.cut(line))
                example_list.append(" ".join(words))
            else:
                line = line.lower()
                words = re.split("\W+", line)
                example_list.append(" ".join(words))

    with open("./data/shijie.txt", "w", encoding='utf8') as f:
        [f.write(ex+"\n") for ex in example_list ]



def train(filename:str):
    model = fasttext.train_unsupervised(filename,  epoch = 10)
    model.save_model("./data/shijie.bin")
    wordlist = model.get_words()

    with open("./data/shijie.vec", "w", encoding='utf8') as f:
        firstline = "%d %d\n"%(len(wordlist), model.get_dimension())
        f.write(firstline)
        for w in wordlist:
            vec = model.get_word_vector(w)
            vecstr= ""
            for val in vec:
                vecstr += " "+str(val)
            f.write(w+" "+vecstr+"\n")
            #print(w)


def cosSim(v1:np.ndarray,v2:np.ndarray):
    return np.dot(v1,v2)/(np.linalg.norm(v1,ord=2)*np.linalg.norm(v2, ord=2))

def test(filename:str):
    model = fasttext.load_model(filename)
    print(model.get_word_id("father"))
    print(model.get_word_id("mother"))
    print(model.get_word_id("wife"))
    print(model.get_word_id("husband"))

    print(model.get_nearest_neighbors("father"))
    print(model.get_analogies("berlin", "germany", "france"))
    print(model.get_analogies("son", "father", "mother"))

#loadData("E:\\DeepLearning\\data\\nlp_data\\平凡的世界.txt", True)
#loadData("E:\\DeepLearning\\data\\nlp_data\\English_corpus.txt", False)
#train("e:/DeepLearning/data/nlp_data/enwik9.txt")
test("./data/shijie.bin")
