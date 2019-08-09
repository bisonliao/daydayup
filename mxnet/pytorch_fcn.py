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
import matplotlib.pyplot as plt

epoches = 500
batchsz = 10
lr = 0.0001 # 随着训练的深入，逐步调小lr，一开始是0.1
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

#测试准确率
def test(model, test_data):
    model.eval()
    s = nn.Softmax(dim=1)
    cnt = 0
    same = 0

    with torch.no_grad(): # 节约内存考虑，关闭梯度
        for images, labels in test_data:
            images = images.to(device="cuda:0")/255
            labels = labels.to(device="cuda:0")
            y = model(images)
            y = y['out']
            y = y.reshape(y.shape[0], y.shape[1], -1)
            y = s(y)
            y = torch.max(y, 1) # type:torch.return_types.max
            labels = labels.reshape(labels.shape[0], -1)
            same = same + (y[1] == labels).sum().to(device='cpu').numpy()
            cnt = cnt + 1
            #torch.cuda.empty_cache()
            if (cnt >= 100):
                break
    return same / batchsz /(cropsz*cropsz)/cnt

# 对样本进行处理，保存为可直观感受的图片
def eval(model, eval_data, e):
    with torch.no_grad(): # 节约内存考虑，关闭梯度
        for images, _ in eval_data:
            images = images.to(device="cuda:0")/255
            yy = model(images)
            yy = yy['out'].to("cpu")
            # shape: batchsz, 21, 40000
            yy = yy.reshape(yy.shape[0], yy.shape[1], -1)  # type:torch.Tensor
            fig = plt.figure(figsize=(50,5)) # width,height
            for i in range(batchsz):
                pic = images[i].to("cpu")#type:torch.Tensor
                pic = (pic * 255 + 128) /255
                pic = pic.reshape((3, -1))
                pic = pic.transpose(1,0)
                pic = pic.reshape((cropsz, cropsz, 3))

                plt.subplot(2, batchsz, i+1)
                plt.imshow(pic)
                plt.axis('off')


                y = yy[i]  # shape:21, 40000
                y = y.transpose(1,0)
                y = y.argmax(1)
                y = y * 255
                pic = torch.zeros((3, cropsz*cropsz), dtype=torch.uint8)

                pic[0] = y
                pic[1] = y
                pic[2] = y
                pic = pic.transpose(1, 0)
                pic = pic.reshape((cropsz, cropsz, 3))
                plt.subplot(2, batchsz, i+batchsz+1)
                plt.imshow(pic)
                plt.axis('off')
            filename="./data/eval_%d.png"%(e)
            fig.savefig(filename)
            break




# 从HDF5文件中读取样本的dataset
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
        #return 100 #尝试对有限的几个训练样本，看看模型会不会过拟合的收敛到它们。结果是ok的，说明模型可以收敛。
        if self.isTrain:
            return self.sampleNumInFile * 50
        else:
            return self.sampleNumInFile * 4


set1 = myDataset()
train_data = dataloader.DataLoader(set1, batchsz, False)# type:dataloader.DataLoader

set1 = myDataset(False)
test_data = dataloader.DataLoader(set1, batchsz, False)# type:dataloader.DataLoader



if compute:
    model = get_net()
    # print(model)
    load(model, 1300)
    # trainer = torch.optim.SGD(model.parameters(), lr,momentum=0.9, weight_decay=0.001)
    trainer = torch.optim.Adam(model.parameters(), lr)
    lossfun = nn.CrossEntropyLoss()

    lossSum = 0
    for e in range(epoches):
        model.train()
        trainer.zero_grad()

        for images, labels in train_data:
            minbatch = minbatch + 1
            images = images.to(device="cuda:0")/255# type:torch.tensor()
            labels = labels.to(device="cuda:0")
            y = model(images)
            y = y['out']
            labels = labels.reshape(batchsz, cropsz, cropsz)
            L = lossfun(y, labels)
            trainer.zero_grad()
            L.backward()
            trainer.step()

            lossSum= lossSum + L.to("cpu").data.numpy()
            if minbatch % 10 == 0:
                print(e, " ",minbatch, " ", lossSum / 10)
                lossSum = 0
            '''if minbatch % 50 == 0:
                for p in trainer.param_groups:
                    p['lr'] *= 0.5'''
            if minbatch % 100 == 0:
                save(model, minbatch)
                eval(model, test_data, minbatch)
                model.train()


else:
    model = get_net()
    load(model, 1300)
    eval(model, test_data, 1)



