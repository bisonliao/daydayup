'''
这个例子是对头条新闻的标题+摘要进行分类（文化、汽车、养生等10多个类）
因为是中文，所以用到了jieba分词和fastText词向量。 top3的准确率可以到99%，top1的准确率可以到97%：
373, 0.1358045231, 0.99, 0.97
374, 0.1353501165, 0.99, 0.97
375, 0.1348878791, 0.99, 0.97
376, 0.1344397208, 0.99, 0.97
377, 0.1339880961, 0.99, 0.97

常用的词向量库有gensim, GloVe, fastText，Word2Vec 等。
fastText is a library for efficient learning of word representations and sentence classification.
fastText既可以获得词向量，也可以对句子进行分类
这个例子里面，只用fastText的词向量功能和fastText的训练好的词向量模型，下载地址：https://fasttext.cc/docs/en/pretrained-vectors.html
没有用fastText的分类功能，是我自己用pytorch实现的分类器。
pip3 install fasttext安装版本有问题（2019.9），还是要从官方的github下的才好。要不然get_nearest_neighbors这么关键的特性都不支持

词向量的使用确实对该分类模型的收敛有明显帮助：
1、当我错误的把英文词向量用于中文的时候，准确率低近20个百分点，且收敛很慢
2、每个单词使用该单词的32B大小的sha256而不是300b大小的词向量表示，准确率收敛很慢
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

batchsz = 1000
hidden_len = 64
emb_sz = 300
seq_len = 40 #每个样本分词个数最大为40
lr = 0.005
epochs = 1000
use_rnn = False #使用rnn还是textCNN网络

def getHash(x):
    m = hashlib.sha256()
    m.update(bytes(x, "utf8"))
    return torch.tensor([int(d)/255 for d in m.digest()])

def loadData():
    all_categories=dict()
    example_list = list()
    ft = fasttext.load_model("E:\\DeepLearning\\data\\fastText_model\\cc.zh.300.bin") # 词向量模型
    with open("E:\\DeepLearning\\data\\nlp_data\\头条新闻分类.txt", "r", encoding='utf8') as f:
        for line in f:
            line = line[:-1]
            fields = line.split("_!_")
            if all_categories.__contains__(fields[2]):
                label = all_categories[fields[2]]
            else:
                label = len(all_categories.keys())
                all_categories[fields[2]] = label

            words = list(jieba.cut(fields[3]+","+fields[4]))
            if len(words) > seq_len:
                #print("sentence is too long, please change the seq_len variable!\n")
                words = words[:seq_len]
             # 因为fastText训练好的模型很占用内存，所以训练样本直接放在gpu的内存里了。
            one_example = torch.zeros((seq_len, emb_sz)).to("cuda:0")
            for i in range(len(words)):
                w = words[i]
                one_example[i] = torch.tensor(ft.get_word_vector(w)).to("cuda:0")
                #one_example[i] = getHash(w).to("cuda:0")

            example_list.append(( one_example,  label))
            if len(example_list) > 30000:
                break
    random.shuffle(example_list)
    #del ft
    return example_list, all_categories

examples, categories = loadData()
print("dataset is ready!")

class myDataset(dataset.Dataset):
    train = list() # static variables
    test = list()

    def __init__(self, source:list, isTrain=True):
        super(myDataset, self).__init__()
        self.isTrain = isTrain

        if len(myDataset.train) < 10:
            for t in source:
                r = random.randint(0, 10)
                if r > 8:
                    self.test.append(t)
                else:
                    self.train.append(t)
            print("test size:", len(myDataset.test))
            print("train size:", len(myDataset.train))


    def __getitem__(self, index):
        if self.isTrain:
            return myDataset.train[index][0].to(dtype=torch.float32)/255, torch.tensor(myDataset.train[index][1], dtype=torch.long)
        else:
            return myDataset.test[index][0].to(dtype=torch.float32) / 255, torch.tensor(myDataset.test[index][1],
                                                                                    dtype=torch.long)

    def __len__(self):
        if self.isTrain:
            return len(myDataset.train)
        else:
            return len(myDataset.test)

set1 = myDataset(examples)
train_data = dataloader.DataLoader(set1, batchsz, False, drop_last=True)# type:dataloader.DataLoader

set1 = myDataset(examples,False)
test_data = dataloader.DataLoader(set1, batchsz, False, drop_last=True)# type:dataloader.DataLoader

class TextCNN(torch.nn.Module):
    def __init__(self, output_size):
        super(TextCNN, self).__init__()
        self.chnnum = 8
        self.conv1 = nn.Conv1d(emb_sz, self.chnnum, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(emb_sz, self.chnnum, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(emb_sz, self.chnnum, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(3, padding=0, stride=3)
        self.fc = nn.Linear(self.chnnum * (seq_len//3)*3, output_size)


    def forward(self, x:torch.Tensor):
        x = x.transpose(2,1)
        x1 = nn.functional.relu(self.conv1(x)) # input shape: (batch_sz, emb_sz(in_chann), seq_len)
        x2 = nn.functional.relu(self.conv2(x))
        x3 = nn.functional.relu(self.conv3(x))

        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        x3 = x3.reshape(x3.shape[0], -1)
        x = torch.cat([x1, x2, x3 ], dim=1)
        x = self.fc(x)
        return x

model = TextCNN(len(categories.keys()))
model = model.to("cuda:0") #type:nn.Module
criterion = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(model.parameters(), lr)

def eval():
    model.eval()
    right3 = 0
    right1 = 0
    cnt = 0
    s = nn.Softmax(dim = 1)
    for inputs, labels in test_data:
        inputs = inputs.to("cuda:0")  # type:torch.Tensor
        labels = labels.to("cuda:0")  # type:torch.Tensor
        output,_ = callmodel(inputs, labels)
        output = s(output)
        top3 = output.topk(3, dim = 1).indices
        for i in range(output.shape[0]):
            if labels[i].item() in top3[i].cpu().tolist():
                right3 += 1

        top1 = output.topk(1, dim=1).indices
        for i in range(output.shape[0]):
            if labels[i].item() in top1[i].cpu().tolist():
                right1 += 1

        cnt += 1
    return right3 / (cnt * batchsz), right1 / (cnt * batchsz)

class staticVars:
    printFirstLoss = True

def callmodel(inputs, labels):

    if use_rnn:
        # (seq_len, batch_sz, emb_sz)
        inputs = inputs.transpose(1, 0)
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], -1))
        hidden = model.initHidden()
        model.zero_grad()
        for i in range(inputs.shape[0]): # inputs shape: (seq_len, batchsz, input_len)
            output, hidden = model(inputs[i], hidden)
    else:
        # (batch_sz, seq_len, emb_sz)
        model.zero_grad()
        output = model(inputs)

    loss = criterion(output, labels)
    loss.backward()

    trainer.step()
    if staticVars.printFirstLoss: # print the loss value of the first iter
        print("first loss value:", loss.item())
        staticVars.printFirstLoss = False

    return output, loss.item()

def train_loop():
    model.train()
    for e in range(epochs):
        loss = 0
        cnt = 0
        for inputs, labels in train_data:
            inputs = inputs.to("cuda:0") # type:torch.Tensor
            labels = labels.to("cuda:0") # type:torch.Tensor

            _,L = callmodel(inputs, labels)
            loss += L
            cnt += 1
        acc3, acc1 = eval()
        model.train()
        print("%d, %.010f, %.2f, %.2f"%(e, loss/cnt,  acc3, acc1))


train_loop()

