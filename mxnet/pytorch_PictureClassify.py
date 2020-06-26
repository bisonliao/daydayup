# -*- coding: UTF-8 -*-
'''
denseNet, 使用cifa-10数据集合进行训练
top1准确率达到75%， top3可以达到94%
因为cifa的图片很小，33X33，10分类情况下这个准确率应该靠谱

vgg16, 使用cifa-10数据集合进行训练
top1准确率只能到72%，top3达到93.5%
loss在每个样本平均0.0003左右不再收敛。（有对lr反复调整尝试）

网上有人用ResNet准确率能到95%，不知道是怎么做到的。
'''

from torchvision import models
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy
import torch.nn as nn


batchsz=512
device ='cuda:0'
compute=True
lr=0.001
epochnm=100
pretrained='./dense_cifa_25.tar'
#pretrained=''

def show_img(img:torch.Tensor):
    img_toshow = img  # type:torch.Tensor
    img_toshow = img_toshow.numpy()  # type:numpy.ndarray
    img_toshow = numpy.uint8(img_toshow * 255)
    img_toshow = img_toshow.transpose(1, 2, 0)
    plt.imshow(img_toshow)
    plt.show()




transform = transforms.Compose( [transforms.ToTensor()]) #不需要normalize了，这样读出来就是0-1之间的像素值

trainset = torchvision.datasets.CIFAR10(root='E:/DeepLearning/data/image_data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsz,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='E:/DeepLearning/data/image_data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsz,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''
for imgs,labels in trainloader:
    #imgs = imgs.to(device)
    for i in range(4):
        show_img(imgs[i])
    print([classes[labels[j].item()] for j in range(4) ])

    break;
'''

# 用测试集对模型进行校验
def test(model, test_data):
    model.eval()

    total = 0
    same = 0
    same3 = 0
    for inputs, labels in test_data:
        inputs = inputs.to(device="cuda:0")
        labels = labels.to(device="cuda:0")# type:torch.Tensor
        y = model(inputs)
        #top1的准确率
        pred = torch.max(y, 1) # type:torch.return_types.max
        same += (pred[1] == labels).sum().to(device='cpu').numpy()
        #top3的准确率
        _, pred = y.topk(3, 1)#type:torch.Tensor
        sample_num = labels.shape[0]
        for i in range(sample_num):
            if pred[i].tolist().__contains__(labels[i]):
                same3 += 1

        total += batchsz
        if total > 1000:
            break
    return (same / total, same3/total)


# 训练
if compute:
    print("start trainning...")
    model = torchvision.models.densenet121(num_classes=10).to("cuda:0") #type:nn.Module
    trainer = torch.optim.Adam(model.parameters(), lr, weight_decay=0.001) #type:torch.optim
    if len(pretrained)>0: # load from a pretrained file
        mdict, tdict = torch.load(pretrained)
        model.load_state_dict(mdict)
        trainer.load_state_dict(tdict)
        for p in trainer.param_groups:
            print("lr:", p['lr'])

    lossfun = nn.CrossEntropyLoss()
    loss_sum = 0
    minbatch = 1

    for e in range(epochnm):
        model.train()

        for inputs, labels in trainloader:
            inputs = inputs.to(device="cuda:0")  # type:torch.tensor()
            labels = labels.to(device="cuda:0")
            y = model(inputs)

            #L = lossfun(y, labels)
            L = nn.CrossEntropyLoss().__call__(y, labels)
            trainer.zero_grad()
            L.backward()
            trainer.step()

            loss_sum += L.to("cpu").data.numpy()
            minbatch += 1
            if (minbatch % 10) == 0:
                print(e, "loss:", loss_sum)
                loss_sum = 0

        acc, acc3 = test(model, testloader)
        print(e, ">>>>>>acc:", acc, acc3)
        model.train()
        torch.save((model.state_dict(), trainer.state_dict()), "./dense_cifa_%d_%d.tar"%(e, int(acc*100) ))
        # decrease lr gradually
        if (e % 5) == 0 and e > 0:
            for p in trainer.param_groups:
                print("lr:", p['lr'])
                p['lr'] = p['lr'] / 2
