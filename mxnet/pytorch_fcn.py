'''
use pytorch to train a model that can do picture Semantic Segmentation
'''
import torch
from torchvision import models
import torch.utils
import torch.utils.model_zoo
import torch.hub
import numpy as np
import torch.optim
import h5py
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt


epoches = 500
batchsz = 8
lr = 0.0001  # lr很关键，太大了会完全不行。这个值刚好
compute = True
use_zone_model = False #是否使用model zone里预先定义的模型结构，还是自己手写的
minbatch = 0  # min batch id
cropsz = 200

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


#网友实现的FCN，vgg16+deconv，结构比较清晰
class FCN8s(nn.Module):

    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        #初始化参数
        self._initialize_weights()
        vgg16 = models.vgg16(True)
        self.copy_params_from_vgg16(vgg16)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))



def save(model, e):
    file = './data/pytorch_fcn_%d.pk' % (e)
    torch.save(model.state_dict(), file)


def load(model, e):
    file = './data/pytorch_fcn_%d.pk' % (e)
    model.load_state_dict(torch.load(file))
    global minbatch
    minbatch = e + 1


def get_net():
    if not use_zone_model:
        return FCN8s(n_class=2).to("cuda:0")
    else:
        model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=False, force_reload=False)
        # modify the last layer to 2 classes
        model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
       # model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

        return model.to("cuda:0")




# 测试准确率
def test(model, test_data):
    model.eval()
    s = nn.Softmax(dim=1)
    cnt = 0
    same = 0

    with torch.no_grad():  # 节约内存考虑，关闭梯度
        for images, labels in test_data:
            images = images.to(device="cuda:0") / 255
            labels = labels.to(device="cuda:0")
            y = model(images)
            if use_zone_model:
                y = y['out']
            y = y.reshape(y.shape[0], y.shape[1], -1)
            y = s(y)
            y = torch.max(y, 1)  # type:torch.return_types.max
            labels = labels.reshape(labels.shape[0], -1)
            same = same + (y[1] == labels).sum().to(device='cpu').numpy()
            cnt = cnt + 1
            # torch.cuda.empty_cache()
            if (cnt >= 100):
                break
    return same / batchsz / (cropsz * cropsz) / cnt


# 对样本进行处理，保存为可直观感受的图片
def eval(model, eval_data, e):
    model.eval()
    with torch.no_grad():  # 节约内存考虑，关闭梯度
        for images, _ in eval_data:
            images = images.to(device="cuda:0") / 255
            yy = model(images)
            if use_zone_model:
                yy = yy['out']
            yy = yy.to("cpu")
            print("yy values:", yy.max(), yy.min(), yy.mean())
            # shape: batchsz, 21, 40000
            yy = yy.reshape(yy.shape[0], yy.shape[1], -1)  # type:torch.Tensor
            fig = plt.figure(figsize=(50, 5))  # width,height
            for i in range(batchsz):
                pic = images[i].to("cpu")  # type:torch.Tensor
                pic = (pic * 255 + 128) / 255
                pic = pic.reshape((3, -1))
                pic = pic.transpose(1, 0)
                pic = pic.reshape((cropsz, cropsz, 3))

                plt.subplot(2, batchsz, i + 1)
                plt.imshow(pic)
                plt.axis('off')

                y = yy[i]  # shape:21, 40000
                y = y.transpose(1, 0)
                y = y.argmax(1)
                y = y * 255
                pic = torch.zeros((3, cropsz * cropsz), dtype=torch.uint8)

                pic[0] = y
                pic[1] = y
                pic[2] = y
                pic = pic.transpose(1, 0)
                pic = pic.reshape((cropsz, cropsz, 3))
                plt.subplot(2, batchsz, i + batchsz + 1)
                plt.imshow(pic)
                plt.axis('off')
            filename = "./data/eval_%d.png" % (e)
            fig.savefig(filename)
            break

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)

    func = nn.LogSoftmax(dim=1)

    log_p = func(input)

    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]

    func = nn.NLLLoss( weight=weight, reduction='sum')
    loss = func(log_p, target)
    if size_average:
        loss /= mask.data.sum()
    return loss

# 从HDF5文件中读取样本的dataset
class myDataset(dataset.Dataset):
    def __init__(self, isTrain=True):
        super()
        self.isTrain = isTrain
        self.path = "E:\\DeepLearning\\data\\image_data\\coco\\HDF5"
        self.sampleNumInFile = 1000

    def __getitem__(self, index):
        if self.isTrain:
            fileIndex = index // self.sampleNumInFile + 100
            recIndex = index % self.sampleNumInFile
        else:
            fileIndex = index // self.sampleNumInFile + 150
            recIndex = index % self.sampleNumInFile
        filename = "%s\\train_%d.h5" % (self.path, fileIndex)
        f = h5py.File(filename, "r")
        data = f["data"]
        d = data[recIndex]
        label = f["label"]
        l = label[recIndex]
        return torch.tensor(d, dtype=torch.float32), torch.tensor(l, dtype=torch.long)

    def __len__(self):
        # return 100 #尝试对有限的几个训练样本，看看模型会不会过拟合的收敛到它们。结果是ok的，说明模型可以收敛。
        if self.isTrain:
            return self.sampleNumInFile * 50
        else:
            return self.sampleNumInFile * 4


set1 = myDataset()
train_data = dataloader.DataLoader(set1, batchsz, False)  # type:dataloader.DataLoader

set1 = myDataset(False)
test_data = dataloader.DataLoader(set1, batchsz, False)  # type:dataloader.DataLoader

if compute:
    model = get_net()
    # print(model)
    #load(model, 1300)
    # trainer = torch.optim.SGD(model.parameters(), lr,momentum=0.9, weight_decay=0.001)
    trainer = torch.optim.Adam(model.parameters(), lr)
    #lossfun = nn.CrossEntropyLoss() #这样也可以的
    lossfun = cross_entropy2d

    lossSum = 0
    for e in range(epoches):
        model.train()
        trainer.zero_grad()

        for images, labels in train_data:
            minbatch = minbatch + 1
            images = images.to(device="cuda:0") / 255  # type:torch.tensor()
            labels = labels.to(device="cuda:0")
            y = model(images)
            if use_zone_model:
                y = y['out']
            labels = labels.reshape(batchsz, cropsz, cropsz)
            L = lossfun(y, labels)
            trainer.zero_grad()
            L.backward()
            trainer.step()

            lossSum = lossSum + L.to("cpu").data.numpy()
            if minbatch % 10 == 0:
                print(e, " ", minbatch, " ", lossSum / 10)
                lossSum = 0
            '''if minbatch % 50 == 0:
                for p in trainer.param_groups:
                    p['lr'] *= 0.5'''
            if minbatch % 100 == 0:
                save(model, minbatch)
                eval(model, test_data, minbatch)
                model.train()

        for p in trainer.param_groups:
            p['lr'] *= 0.1


else:
    model = get_net()
    load(model, 1300)
    eval(model, test_data, 1)
