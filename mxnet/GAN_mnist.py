'''
使用GAN网络，生成mnist这样的手写图片，或者CIFAR这样的图片
'''

import mxnet as mx
from mxnet import nd
from mxnet import gluon, autograd
from mxnet.gluon import Block
import os
import random
import logging
import argparse
import numpy as np
import math
import mxnet.gluon.nn as nn

import imageio

EPSILON = 1e-08
POWER_ITERATION = 1
########################################################
# args
BATCH_SIZE = 100
Z_DIM = 100
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
BETA = 0.5
OUTPUT_DIR = "./data"
CTX = mx.gpu(0)
CLIP_GRADIENT = 10.0
IMAGE_SIZE = 64

DATA_SOURCE = "cifar"  # cifar or mnist

########################################################
# training samples source

def transform(data, label):
    data = mx.image.imresize(data, IMAGE_SIZE, IMAGE_SIZE)
    if DATA_SOURCE=='cifar':
        data = mx.nd.transpose(data, (2, 0, 1))
    else:
        data = data.reshape(1,IMAGE_SIZE,IMAGE_SIZE)
        data = nd.tile(data, reps=(3,1,1)) # 3 channels BGR
    return data.astype(np.float32)/128-1, label.astype(np.float32)
if DATA_SOURCE=='cifar':
    dataset = gluon.data.vision.CIFAR10('./data', train=True, transform=transform)
else:
    dataset = mx.gluon.data.vision.MNIST(train=True, transform=transform)
train_example_num = len(dataset)
print("train samples number:", train_example_num)
train_data = mx.gluon.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

###########################################################
# build the generator
nc = 3
ngf = 64
netG = nn.Sequential()
with netG.name_scope():
    # input is Z, going into a convolution
    netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 4 x 4
    netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 8 x 8
    netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 16 x 16
    netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 32 x 32
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, use_bias=False))
    netG.add(nn.Activation('tanh'))
    # state size. (nc) x 64 x 64

# build the discriminator
ndf = 64
netD = nn.Sequential()
with netD.name_scope():
    # input is (nc) x 64 x 64
    netD.add(nn.Conv2D(ndf, 4, 2, 1, use_bias=False))
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 32 x 32
    netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 16 x 16
    netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 8 x 8
    netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 4 x 4
    netD.add(nn.Conv2D(1, 4, 1, 0, use_bias=False))
###########################################################
# train
def save_image(data, epoch, image_size, batch_size, output_dir, padding=2):
    """ save image """
    data = data.asnumpy().transpose((0, 2, 3, 1))
    datanp = np.clip(
        (data - np.min(data))*(255.0/(np.max(data) - np.min(data))), 0, 255).astype(np.uint8)
    x_dim = min(8, batch_size)
    y_dim = int(math.ceil(float(batch_size) / x_dim))
    height, width = int(image_size + padding), int(image_size + padding)
    grid = np.zeros((height * y_dim + 1 + padding // 2, width *
                     x_dim + 1 + padding // 2, 3), dtype=np.uint8)
    k = 0
    for y in range(y_dim):
        for x in range(x_dim):
            if k >= batch_size:
                break
            start_y = y * height + 1 + padding // 2
            end_y = start_y + height - padding
            start_x = x * width + 1 + padding // 2
            end_x = start_x + width - padding
            np.copyto(grid[start_y:end_y, start_x:end_x, :], datanp[k])
            k += 1
    imageio.imwrite(
        '{}/fake_samples_epoch_{}.png'.format(output_dir, epoch), grid)


def facc(label, pred):
    """ evaluate accuracy """
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


# setting
mx.random.seed(random.randint(1, 10000))
logging.basicConfig(level=logging.DEBUG)




# get model
g_net = netG
d_net = netD

# define loss function
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# initialization
g_net.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
d_net.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
g_trainer = gluon.Trainer(
    g_net.collect_params(), 'Adam', {'learning_rate': LEARNING_RATE, 'beta1': BETA, 'clip_gradient': CLIP_GRADIENT})
d_trainer = gluon.Trainer(
    d_net.collect_params(), 'Adam', {'learning_rate': LEARNING_RATE, 'beta1': BETA, 'clip_gradient': CLIP_GRADIENT})
g_net.collect_params().zero_grad()
d_net.collect_params().zero_grad()
# define evaluation metric
metric = mx.metric.CustomMetric(facc)
# initialize labels
real_label = nd.ones(BATCH_SIZE, CTX)
fake_label = nd.zeros(BATCH_SIZE, CTX)

for epoch in range(NUM_EPOCHS):
    for i, (d, _) in enumerate(train_data):
        # update D
        data = d.as_in_context(CTX)
        noise = nd.normal(loc=0, scale=1, shape=(BATCH_SIZE, Z_DIM, 1, 1), ctx=CTX)
        with autograd.record():
            # train with real image
            output = d_net(data).reshape((-1, 1))
            errD_real = loss(output, real_label)
            metric.update([real_label, ], [output, ])

            # train with fake image
            fake_image = g_net(noise)
            output = d_net(fake_image.detach()).reshape((-1, 1))
            errD_fake = loss(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([fake_label, ], [output, ])

        d_trainer.step(BATCH_SIZE)
        # update G
        with autograd.record():
            fake_image = g_net(noise)
            output = d_net(fake_image).reshape(-1, 1)
            errG = loss(output, real_label)
            errG.backward()

        g_trainer.step(BATCH_SIZE)

        # print log infomation every 100 batches
        if i % 200 == 0:
            name, acc = metric.get()
            logging.info('discriminator loss = %f, generator loss = %f, \
                          binary training acc = %f at iter %d epoch %d',
                         nd.mean(errD).asscalar(), nd.mean(errG).asscalar(), acc, i, epoch)
        if i == 0:
            if epoch == 0 :
                save_image(data, epoch, IMAGE_SIZE, BATCH_SIZE, OUTPUT_DIR)
            else:
                save_image(fake_image, epoch, IMAGE_SIZE, BATCH_SIZE, OUTPUT_DIR)

    metric.reset()