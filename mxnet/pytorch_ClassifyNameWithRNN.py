'''
使用rnn对人名进行分类，分出是哪个国家的名字
18个国家（分类），top3的准确率只有78%， top1的准确率只有50%
以为使用rnn的姿势不对，改TextCNN网络，也是一样的结果。应该是名字本身比较短，本身区分度不高导致。
另外一个例子里用TextCNN识别二分类问题，可以到96%的准确率的。
数据来源：https://download.pytorch.org/tutorial/data.zip
'''
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



batchsz = 1000
hidden_len = 64
input_len = 1 #rnn每次输入一个字符的byte
seq_len = 10 #一个名字长度统一为seq_len
lr = 0.005
epochs = 1000
use_rnn = True #使用rnn还是textCNN网络




def loadData():
    category_idx = 0
    all_categories=dict()
    example_list = list()
    for filename in glob.glob('E:\\DeepLearning\\data\\name_classify\\names\\*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories[category_idx] = category

        with open(filename, "rb") as f:
            for line in f:
                line = line[:-1]
                if len(line) > seq_len:
                    print("name is too long, please change the seq_len variable!\n")
                    line = line[:seq_len]
                while len(line) < seq_len:
                    line = line + b'\0'
                '''one_example = torch.zeros((seq_len, input_len))
                for i in range(seq_len):
                    one_hot = torch.zeros((input_len))
                    one_hot[ line[i] ] = 1
                    one_hot[0] = 0
                    one_example[i] = one_hot
                example_list.append((one_example, category_idx))'''
                example_list.append(( torch.tensor([b for b in line]),  category_idx))
        category_idx += 1
    random.shuffle(example_list)
    return example_list, all_categories

examples, categories = loadData()

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


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)


    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) #  (batchsz, input_len)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return torch.zeros(batchsz, self.hidden_size).to("cuda:0")

class TextCNN(torch.nn.Module):
    def __init__(self,input_size, output_size):
        super(TextCNN, self).__init__()
        self.chnnum = 8
        self.conv1 = nn.Conv1d(1, self.chnnum, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(1, self.chnnum, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(1, self.chnnum, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(3, padding=1)
        self.pool2 = nn.MaxPool1d(3, padding=1)
        self.pool3 = nn.MaxPool1d(3, padding=1)
        self.fc = nn.Linear(96, output_size)


    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(x))
        x3 = nn.functional.relu(self.conv3(x))

        x1 = self.pool1(x1)
        x2 = self.pool1(x2)
        x3 = self.pool1(x3)


        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        x3 = x3.reshape(x3.shape[0], -1)
        x = torch.cat([x1, x2, x3 ], dim=1)
        x = self.fc(x)
        return x
if use_rnn:
    model = RNN(input_len, hidden_len, len(categories.keys()))
else:
    model = TextCNN(seq_len, len(categories.keys()))
model = model.to("cuda:0") #type:nn.Module
criterion = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(model.parameters(), lr)
#trainer = torch.optim.SGD(model.parameters(), lr)

def eval():
    model.eval()
    right = 0
    cnt = 0
    s = nn.Softmax(dim = 1)
    for inputs, labels in test_data:
        inputs = inputs.to("cuda:0")  # type:torch.Tensor
        labels = labels.to("cuda:0")  # type:torch.Tensor
        output,_ = callmodel(inputs, labels)
        output = s(output)
        output = output.topk(3, dim = 1).indices
        for i in range(output.shape[0]):
            if labels[i].item() in output[i].cpu().tolist():
                right += 1

        cnt += 1
    return right / (cnt * batchsz)

class staticVars:
    printFirstLoss = True

def callmodel(inputs, labels):

    if use_rnn:
        inputs = inputs.transpose(1, 0)
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], -1))
        hidden = model.initHidden()
        model.zero_grad()
        for i in range(inputs.shape[0]): # inputs shape: (seq_len, batchsz, input_len)
            output, hidden = model(inputs[i], hidden)
    else:
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
        print("%d, %.010f, %.2f"%(e, loss/cnt,  eval()))
        model.train()

train_loop()



