'''
use pytorch to train a model that can do picture Semantic Segmentation
'''
import torch
import torchvision
import torch.utils
import torch.utils.model_zoo
import torch.hub
import numpy as np
import torch.optim
import h5py
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torch.nn as nn

epoches = 500
batchsz = 10
lr = 0.1
compute = True
minbatch = 0  # min batch id
cropsz = 200

def save(model, e):
    file = './data/pytorch_fcn_%d.pk'%(e)
    torch.save(model.state_dict(), file)

def load(model, e):
    file = './data/pytorch_fcn_%d.pk' % (e)
    model.load_state_dict(torch.load(file))
    global  minbatch
    minbatch = e + 1

def get_net():
    model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=False, force_reload=False)
    return model.to("cuda:0")

def test(model, test_data):
    model.eval()
    s = nn.Softmax(dim=1)
    for images, labels in test_data:
        images = images.to(device="cuda:0")
        labels = labels.to(device="cuda:0")
        y = model(images)
        y = y['out']
        y = y.reshape(y.shape[0], y.shape[1], -1)
        y = s(y)
        y = torch.max(y, 1) # type:torch.return_types.max
        labels = labels.reshape(labels.shape[0], -1)
        same = (y[1] == labels).sum()
        return same.to(device='cpu').numpy() / batchsz /(cropsz*cropsz)

class myDataset(dataset.Dataset):
    def __init__(self, isTrain=True):
        super()
        self.isTrain = isTrain
        self.path = "E:\\DeepLearning\\data\\coco\\HDF5"
        self.sampleNumInFile = 1000

    def __getitem__(self, index):
        if self.isTrain:
            fileIndex = index // self.sampleNumInFile + 100
            recIndex = index % self.sampleNumInFile
        else:
            fileIndex = index // self.sampleNumInFile + 150
            recIndex = index % self.sampleNumInFile
        filename = "%s\\train_%d.h5"%(self.path, fileIndex)
        f = h5py.File(filename, "r")
        data = f["data"]
        d = data[recIndex]
        label = f["label"]
        l = label[recIndex]
        return torch.tensor(d, dtype=torch.float32), torch.tensor(l, dtype=torch.long)


    def __len__(self):
        if self.isTrain:
            return self.sampleNumInFile * 50
        else:
            return self.sampleNumInFile * 4


set1 = myDataset()
train_data = dataloader.DataLoader(set1, batchsz, False)# type:dataloader.DataLoader

set1 = myDataset(False)
test_data = dataloader.DataLoader(set1, batchsz, False)# type:dataloader.DataLoader

model = get_net()
print(model)
load(model, 200)
trainer = torch.optim.SGD(model.parameters(), lr)
lossfun = nn.CrossEntropyLoss()

if compute:
    lossSum = 0
    for e in range(epoches):
        model.train()
        trainer.zero_grad()

        for images, labels in train_data:
            minbatch = minbatch + 1
            images = images.to(device="cuda:0")# type:torch.tensor()
            labels = labels.to(device="cuda:0")
            y = model(images)
            y = y['out']
            labels = labels.reshape(batchsz, cropsz, cropsz)
            L = lossfun(y, labels)
            L.backward()
            lossSum= lossSum + L.to("cpu").data.numpy()
            if minbatch % 10 == 0:
                trainer.step()
                trainer.zero_grad()
                print(minbatch, ":", lossSum / 10)
                lossSum = 0
            if minbatch % 200 == 0:
                save(model, minbatch)
                print("test acc:"'', test(model, test_data))


else:
    model = get_net()
    load(model, 10)



