'''
Use fastText to train a model that can classify the text.
These text are  toutiao news title, each has a type label, for example :
没钱 没 资源 如何 改变命运 ？ , __label__news_finance


the train dataset has more than 300000 records, but  the top1 accuracy is only 88%,
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



def loadData():
    example_list = list()
    with open("E:\\DeepLearning\\data\\nlp_data\\头条新闻分类.txt", "r", encoding='utf8') as f:
        for line in f:
            line = line[:-1]
            fields = line.split("_!_")
            # 分词后，用空格隔开句子中的单词
            words = list(jieba.cut(fields[3]+","+fields[4]))
            one_example = "%s __label__%s\n"%(" ".join(words), fields[2])
            example_list.append(one_example)
    random.shuffle(example_list)
    sep = int(len(example_list) * 0.8)
    with open("./data/fastTextClassify_train.txt", "w", encoding='utf8') as f:
        [f.write(ex) for ex in example_list[:sep] ]
    with open("./data/fastTextClassify_valid.txt", "w", encoding='utf8') as f:
        [f.write(ex) for ex in example_list[sep:] ]
    return example_list[:sep], example_list[sep:]

if False:
    trainDataset, testDataset = loadData()
    with open("./data/testDataset.pl", "wb") as f:
        pickle.dump(testDataset, f)
else:
    with open("./data/testDataset.pl", "rb") as f:
        testDataset = pickle.load(f)

# 1.Using pretrained wordembeding file is not helpful for accuracy.
#model = fasttext.train_supervised(input = "./data/fastTextClassify_train.txt", epoch=100,dim=300,pretrainedVectors ="E:\\DeepLearning\\data\\fastText_model\\cc.zh.300.vec" )
model = fasttext.train_supervised(input = "./data/fastTextClassify_train.txt", epoch=100, wordNgrams=2)
print(model.test("./data/fastTextClassify_valid.txt", k=1))
print(model.test("./data/fastTextClassify_valid.txt", k=3)) #怎么top3的准确率还低些哦？ fastText的top3的意义和我们平时理解的不一样：The precision is the number of correct labels among the labels predicted by fastText.
#model.save_model("./data/fastTextClassify.bin")


def eval():
    acc = 0
    cnt = 0
    for oneExample in testDataset:
        oneExample = oneExample[:-1]
        cnt += 1
        fields = oneExample.split("__label__")
        result = model.predict(fields[0], k = 3)[0]
        if "__label__"+fields[1] in result:
            acc += 1
    return acc / cnt
print(eval()) #按照我的理解的统计，准确率可以到97%

