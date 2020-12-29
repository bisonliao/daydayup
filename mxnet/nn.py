# -*- coding:utf-8 -*-

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

batchsz = 100
lr = 0.01
inputsz = 40
epochnm = 30
minbatch = 0
compute = True



class myDataset(dataset.Dataset):
    def __init__(self, isTrain=True):
        super(myDataset, self).__init__()
        self.isTrain = isTrain
        m = 1000
        X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

        self.train = list()
        self.test = list()

        for i in range(len(X_moons)):  # type: int
            #t = (torch.tensor([ float( (X_moons[i][j]+5) / 10.0) for j in range(len(X_moons[i]))]), y_moons[i]) #归一化反而效果很差，搞不太懂
            t = (torch.tensor([float(X_moons[i][j] ) for j in range(len(X_moons[i]))]), y_moons[i])
            self.train.append(t)
            r = random.randint(0, 10)
            if r > 9:
                self.test.append(t)


        random.shuffle(self.test)
        random.shuffle(self.train)
        print("test size:", len(self.test))
        print("train size:", len(self.train))

    def __getitem__(self, index):
        if self.isTrain:
            return self.train[index][0].to(dtype=torch.float32) , torch.tensor(self.train[index][1],
                                                                                    dtype=torch.long)
        else:
            return self.test[index][0].to(dtype=torch.float32) , torch.tensor(self.test[index][1],
                                                                                   dtype=torch.long)

    def __len__(self):
        if self.isTrain:
            return len(self.train)
        else:
            return len(self.test)


set1 = myDataset()
train_data = dataloader.DataLoader(set1, batchsz, False)  # type:dataloader.DataLoader

set1 = myDataset(False)
test_data = dataloader.DataLoader(set1, batchsz, False)  # type:dataloader.DataLoader


# 简单的神经网络、
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)


    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x





# 用测试集对模型进行校验
def test(model, test_data):
    model.eval()
    s = nn.Softmax(dim=1)
    total = 0
    same = 0
    for inputs, labels in test_data:
        y = model(inputs)
        y = s(y)
        y = torch.max(y, 1)  # type:torch.return_types.max
        same += (y[1] == labels).sum().to(device='cpu').numpy()
        total += batchsz
        if total > 1000:
            break
    return same / total


def draw(model, test_data):
    model.eval()
    s = nn.Softmax(dim=1)

    xx = []
    yy = []

    for inputs, labels in test_data:
        y = model(inputs)
        y = s(y)
        y = torch.max(y, 1)  # type:torch.return_types.max
        y = y.indices

        for i in range(len(inputs)):
            if y[i] == 0:
                xx.append(inputs[i,0].numpy())
                yy.append(inputs[i,1].numpy())
    plt.scatter(xx, yy, color="r")

    xx = []
    yy = []
    for inputs, labels in test_data:
        y = model(inputs)
        y = s(y)
        y = torch.max(y, 1)  # type:torch.return_types.max
        y = y.indices

        for i in range(len(inputs)):
            if y[i] == 1:
                xx.append(inputs[i,0].numpy())
                yy.append(inputs[i,1].numpy())
    plt.scatter(xx, yy, color="b")
    plt.show()


# 训练
if compute:
    model = MyModel()
    trainer = torch.optim.Adam(model.parameters(), lr)
    lossfun = nn.CrossEntropyLoss()
    lossSum = 0

    for e in range(epochnm):
        model.train()

        for inputs, labels in train_data:
            minbatch += 1

            y = model(inputs)

            L = lossfun(y, labels)
            trainer.zero_grad()
            L.backward()
            trainer.step()

            lossSum = lossSum + L.data.numpy()
            if minbatch % 10 == 0:
                print(e, " ", minbatch, " ", lossSum / 200)
                lossSum = 0
        print(e, " acc:", test(model, test_data))
        model.train()
    draw(model, train_data)

