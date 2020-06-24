# -*- coding: UTF-8 -*-
'''
计算模型中需要学习的参数的个数
输出：
inception: 13004888
vgg16: 134346581
resnet: 23551061
dense 6975381

除了参数的个数，其实前向传播计算量、后向传播计算量、模型的大小也是我们经常关注的
但后面几个不容易直接统计了

另外有个这样的工具可以统计模型的参数和前向传播计算量：
https://github.com/Lyken17/pytorch-OpCounter
'''

from torchvision import models

inception = models.GoogLeNet()
vgg = models.vgg16(num_classes=21)
res = models.resnet50(num_classes=21)
dense = models.densenet121(num_classes=21)

param_cnt = 0
for p in inception.parameters():
    param_cnt += p.data.nelement()
print("inception:", param_cnt)

param_cnt = 0
for p in vgg.parameters():
    param_cnt += p.data.nelement()
print("vgg16:", param_cnt)

param_cnt = 0
for p in res.parameters():
    param_cnt += p.data.nelement()
print("resnet:", param_cnt)

param_cnt = 0
for p in dense.parameters():
    param_cnt += p.data.nelement()
print("dense", param_cnt)