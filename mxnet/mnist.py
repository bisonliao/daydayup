from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import random
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import time
import datetime

batchsz = 100
lr = 0.0001
epochnm = 1
minbatch = 0
compute = True

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("device=", device)

begin = time.mktime(datetime.datetime.now().timetuple())


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.MNIST('~/MNIST_data/', download=True, train=True, transform=transform)

testset = torchvision.datasets.MNIST('~/MNIST_data/', download=True, train=False, transform=transform)
train_data = dataloader.DataLoader(trainset, batchsz, False)  # type:dataloader.DataLoader
test_data = dataloader.DataLoader(testset, batchsz, False)  # type:dataloader.DataLoader

# 简单的神经网络、
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.fc1 = nn.Linear(50176, 512)
        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = x.reshape(x.shape[0], -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 用测试集对模型进行校验
def test(model, test_data):
    model.eval()
    s = nn.Softmax(dim=1)
    total = 0
    same = 0
    for inputs, labels in test_data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        y = model(inputs)
        y = s(y)
        y = torch.max(y, 1)  # type:torch.return_types.max
        same += (y[1] == labels).sum().to(device='cpu').numpy()
        total += batchsz
        if total > 200:
            break
    return same / total


def draw(model, test_data):
    model.eval()
    s = nn.Softmax(dim=1)


    for inputs, labels in test_data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        y = model(inputs)
        y = s(y)
        y = torch.max(y, 1)  # type:torch.return_types.max
        y = y.indices
        y = y.to("cpu")
        inputs = inputs.to("cpu")
        for i in range(batchsz):
            plt.imshow(inputs[i,0])
            plt.show()
            print(y[i])


# 训练
model = MyModel()
model = model.to(device)
if compute:
    trainer = torch.optim.Adam(model.parameters(), lr)
    lossfun = nn.CrossEntropyLoss()
    lossSum = 0

    for e in range(epochnm):
        model.train()

        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            minbatch += 1

            y = model(inputs)

            L = lossfun(y, labels)
            trainer.zero_grad()
            L.backward()
            trainer.step()

            lossSum = lossSum + L.data.to("cpu").numpy()
            if minbatch % 10 == 0:
                print(e, " ", minbatch, " ", lossSum / 200)
                lossSum = 0
            if (minbatch% 50) == 0:
                print(e, " acc:", test(model, test_data))
        model.train()


end = time.mktime(datetime.datetime.now().timetuple())
print("used seconds:", end-begin)
draw(model, test_data)




