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

########################################################
# training samples source
def transform(data, label):
    #data shape is (28, 28, 1), transfer to (1, 28, 28),
    # because conv need input shape is (N, C, H, W)
    data = mx.image.imresize(data, IMAGE_SIZE, IMAGE_SIZE)
    data = data.reshape(1,IMAGE_SIZE,IMAGE_SIZE)
    data = nd.tile(data, reps=(3,1,1)) # 3 channels BGR
    return data.astype(np.float32)/128-1, label.astype(np.float32)

dataset = mx.gluon.data.vision.MNIST(train=True, transform=transform)
train_example_num = len(dataset)
print("train samples number:", train_example_num)
train_data = mx.gluon.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

###########################################################
# define the model
class SNConv2D(Block):
    """ Customized Conv2D to feed the conv with the weight that we apply spectral normalization """

    def __init__(self, num_filter, kernel_size,
                 strides, padding, in_channels,
                 ctx=mx.cpu(), iterations=1):

        super(SNConv2D, self).__init__()

        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.in_channels = in_channels
        self.iterations = iterations
        self.ctx = ctx

        with self.name_scope():
            # init the weight
            self.weight = self.params.get('weight', shape=(
                num_filter, in_channels, kernel_size, kernel_size))
            self.u = self.params.get(
                'u', init=mx.init.Normal(), shape=(1, num_filter))

    def _spectral_norm(self):
        """ spectral normalization """
        w = self.params.get('weight').data(self.ctx)
        w_mat = nd.reshape(w, [w.shape[0], -1])

        _u = self.u.data(self.ctx)
        _v = None

        for _ in range(POWER_ITERATION):
            _v = nd.L2Normalization(nd.dot(_u, w_mat))
            _u = nd.L2Normalization(nd.dot(_v, w_mat.T))

        sigma = nd.sum(nd.dot(_u, w_mat) * _v)
        if sigma == 0.:
            sigma = EPSILON

        with autograd.pause():
            self.u.set_data(_u)

        return w / sigma

    def forward(self, x):
        # x shape is batch_size x in_channels x height x width
        return nd.Convolution(
            data=x,
            weight=self._spectral_norm(),
            kernel=(self.kernel_size, self.kernel_size),
            pad=(self.padding, self.padding),
            stride=(self.strides, self.strides),
            num_filter=self.num_filter,
            no_bias=True
        )


def get_generator():
    """ construct and return generator """
    g_net = gluon.nn.Sequential()
    with g_net.name_scope():

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=512, kernel_size=4, strides=1, padding=0, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=256, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=128, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=64, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(channels=3, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.Activation('tanh'))

    return g_net


def get_descriptor(ctx):
    """ construct and return descriptor """
    d_net = gluon.nn.Sequential()
    with d_net.name_scope():

        d_net.add(SNConv2D(num_filter=64, kernel_size=4, strides=2, padding=1, in_channels=3, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=128, kernel_size=4, strides=2, padding=1, in_channels=64, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=256, kernel_size=4, strides=2, padding=1, in_channels=128, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=512, kernel_size=4, strides=2, padding=1, in_channels=256, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=1, kernel_size=4, strides=1, padding=0, in_channels=512, ctx=ctx))

    return d_net
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

# create output dir
try:
    os.makedirs(OUTPUT_DIR)
except OSError:
    pass


# get model
g_net = get_generator()
d_net = get_descriptor(CTX)

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
        if i % 100 == 0:
            name, acc = metric.get()
            logging.info('discriminator loss = %f, generator loss = %f, \
                          binary training acc = %f at iter %d epoch %d',
                         nd.mean(errD).asscalar(), nd.mean(errG).asscalar(), acc, i, epoch)
        if i == 0:
            save_image(fake_image, epoch, IMAGE_SIZE, BATCH_SIZE, OUTPUT_DIR)

    metric.reset()