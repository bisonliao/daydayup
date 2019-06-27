'''
训练一个GAN，实现黑白图片上色。 效果不太好！
图片转化为Lab色彩模式进行计算
对于NetG，输入黑白图片，输出彩色照片
对于NetD，输入彩色图片，输出真伪判断。

也尝试用这个网络来修复图片中被抠掉的矩形区域，无论是对coco数据集还是简单的卡通头像，效果都不是很好。
'''
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from datetime import datetime
import time
import logging
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np
import math
import imageio
import re
import random
import cv2

#######################################################################
# arguments:
epochs = 200
batch_size = 10

use_gpu = True
ctx = mx.gpu() if use_gpu else mx.cpu()

lr = 0.0002
beta1 = 0.5
lambda1 = 100
pool_size = 50
reversed = True
#######################################################################
# define dataset
img_wd = 256
img_ht = 256
train_img_path = 'E:\\DeepLearning\\data\\CGAN\\colorize\\train'
val_img_path = 'E:\\DeepLearning\\data\\CGAN\\colorize\\val'


class MyFileDataset(mx.gluon.data.Dataset):
    def __init__(self, path, is_reversed=True):
        super(MyFileDataset, self)
        self.is_reversed = is_reversed
        self.filelist = []
        for path, _, fnames in os.walk(path):
            for fname in fnames:
                if not fname.endswith('.jpg'):
                    continue
                img = os.path.join(path, fname)
                self.filelist.append(img)
        random.shuffle(self.filelist)
        return



    def __getitem__(self, item):
        img = self.filelist[item]
        img = mx.image.imread(img)
        img = mx.image.fixed_crop(img, 0, 0, img_wd, img_ht)
        img = cv2.cvtColor(img.asnumpy().astype('uint8'), cv2.COLOR_BGR2LAB)

        img = nd.array(img,dtype='float32')
        img = img / 255.0
        img = nd.transpose(img, (2, 0, 1))
        i = img[0].reshape(1, img_wd, img_ht)  # L channel
        i = nd.concat(i,    nd.ones(shape=(1, img_wd, img_ht))*0.5,
                            nd.ones(shape=(1, img_wd, img_ht))*0.5, dim=0)
        o = img  # a b channel



        return i, o

    def __len__(self):
        return len(self.filelist)


train_data = gluon.data.DataLoader(MyFileDataset(train_img_path, reversed), batch_size=batch_size, shuffle=False,
                                   last_batch='discard')
val_data = gluon.data.DataLoader(MyFileDataset(val_img_path, reversed), batch_size=batch_size, shuffle=False,
                                 last_batch='discard')


def visualize(img_arr):
    img_arr = img_arr*255.0

    img_arr = img_arr.transpose((1, 2, 0))
    img_arr = cv2.cvtColor(img_arr.astype('uint8').asnumpy(), cv2.COLOR_LAB2BGR)


    plt.imshow(img_arr.astype(np.uint8))
    plt.axis('off')


def preview_train_data():
    for _, (img_in_list, img_out_list) in enumerate(train_data):
        for i in range(4):
            plt.subplot(2, 4, i + 1)
            visualize(img_in_list[i])
            plt.subplot(2, 4, i + 5)
            visualize(img_out_list[i])
        plt.show()
        break


preview_train_data()



#######################################################################
# Define Unet generator skip block
class UnetSkipUnit(HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False,
                 use_dropout=False, use_bias=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)
            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)

            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + decoder
            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2)
                encoder = [en_conv]
                decoder = [de_relu, de_conv, Activation(activation='tanh')]
                model = encoder + [inner_block] + decoder
            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2, use_bias=use_bias)
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + [inner_block] + decoder
            if use_dropout:
                model += [Dropout(rate=0.5)]

            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return F.concat(self.model(x), x, dim=1)


# Define Unet generator
class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=True):
        super(UnetGenerator, self).__init__()

        # Build unet generator structure
        unet = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet = UnetSkipUnit(ngf * 8, ngf * 8, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf * 8, ngf * 4, unet)
        unet = UnetSkipUnit(ngf * 4, ngf * 2, unet)
        unet = UnetSkipUnit(ngf * 2, ngf * 1, unet)
        unet = UnetSkipUnit(ngf, in_channels, unet, outermost=True)

        with self.name_scope():
            self.model = unet

    def hybrid_forward(self, F, x):
        return self.model(x)

# Define the PatchGAN discriminator
class Discriminator(HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False):
        super(Discriminator, self).__init__()

        with self.name_scope():
            self.model = HybridSequential()
            kernel_size = 4
            padding = int(np.ceil((kernel_size - 1) / 2))
            self.model.add(Conv2D(channels=ndf, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=in_channels))
            self.model.add(LeakyReLU(alpha=0.2))

            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=2,
                                      padding=padding, in_channels=ndf * nf_mult_prev,
                                      use_bias=use_bias))
                self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
                self.model.add(LeakyReLU(alpha=0.2))

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult_prev,
                                  use_bias=use_bias))
            self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
            self.model.add(LeakyReLU(alpha=0.2))
            self.model.add(Conv2D(channels=1, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult))
            if use_sigmoid:
                self.model.add(Activation(activation='sigmoid'))

    def hybrid_forward(self, F, x):
        out = self.model(x)
        # print(out)
        return out


#######################################################################
# init network
def param_init(param):
    if param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
    elif param.name.find('batchnorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx)
        # Initialize gamma from normal distribution with mean 1 and std 0.02
        if param.name.find('gamma') != -1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape))


def network_init(net):
    for param in net.collect_params().values():
        param_init(param)


def set_network():
    # Pixel2pixel networks
    netG = UnetGenerator(in_channels=3, num_downs=8)
    netD = Discriminator(in_channels=3)

    # Initialize parameters
    network_init(netG)
    network_init(netD)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    with open("./data/netG.proto", "w") as f:
        print(netG, file=f)
    with open("./data/netD.proto", "w") as f:
        print(netD, file=f)

    return netG, netD, trainerG, trainerD


# dump param periodly，for break and continue
def dump_param(ep):
    filename = "./data/CGAN_netG_%d.param" % (ep)
    netG.save_parameters(filename)
    filename = "./data/CGAN_netD_%d.param" % (ep)
    netD.save_parameters(filename)


# load param from file if there is pretrained param files，for break and continue
def load_param(path):
    for path, _, fnames in os.walk(path):
        maxG = 0
        maxD = 0
        for fname in fnames:
            index_list = re.findall("CGAN_netG_\d+.param", fname)
            index_list2 = re.findall("\d+", fname)
            if len(index_list) > 0 and int(index_list2[0]) > maxG:
                maxG = int(index_list2[0])

            index_list = re.findall("CGAN_netD_\d+.param", fname)
            index_list2 = re.findall("\d+", fname)
            if len(index_list) > 0 and int(index_list2[0]) > maxD:
                maxD = int(index_list2[0])
    if (maxG > 0 and maxD > 0):
        filename = "./data/CGAN_netG_%d.param" % (maxG)
        print("load net params from ", filename)
        netG.load_parameters(filename, ctx=ctx)
        filename = "./data/CGAN_netD_%d.param" % (maxD)
        netD.load_parameters(filename, ctx=ctx)
        print("load net params from ", filename)
        return maxD + 1
    else:
        return 0


# Loss
GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
L1_loss = gluon.loss.L1Loss()

netG, netD, trainerG, trainerD = set_network()
start_epoch = load_param("./data/")  # if there are saved param files, load params from file


#######################################################################
# We use history image pool to help discriminator memorize history errors
# instead of just comparing current real input and fake output.
# WHY ？
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        ret_imgs = []
        for i in range(images.shape[0]):
            image = nd.expand_dims(images[i], axis=0)
            if self.num_imgs < self.pool_size:  # 历史库还有空位，就填充到历史库
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                ret_imgs.append(image)
            else:
                p = nd.random_uniform(0, 1, shape=(1,)).asscalar()
                if p > 0.5:  # 一定概率取历史库，且更新历史库
                    random_id = nd.random_uniform(0, self.pool_size - 1, shape=(1,)).astype(np.uint8).asscalar()
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    ret_imgs.append(tmp)
                else:  # 一定概率取当期的数据
                    ret_imgs.append(image)
        ret_imgs = nd.concat(*ret_imgs, dim=0)
        return ret_imgs


def test(ep, batch_num=1):
    for i, (real_in, real_out) in enumerate(val_data):
        real_in = real_in.as_in_context(ctx)
        real_out = real_out.as_in_context(ctx)

        fake_out = netG(real_in)
        save_image(nd.concat(real_out, fake_out, dim=0), ep+i, img_wd, "./data")
        if i >= batch_num:
            break


#######################################################################
# train network
def save_image(data, epoch, image_size, output_dir, padding=2):
    """ save image """
    img_num = data.shape[0]
    data = data.asnumpy().transpose((0, 2, 3, 1))
    datanp = (data * 255).astype(np.uint8)
    for k in range(img_num):
        datanp[k] = cv2.cvtColor(datanp[k], cv2.COLOR_LAB2BGR)
    x_dim = 10
    y_dim = int(math.ceil(float(img_num) / x_dim))
    height, width = int(image_size + padding), int(image_size + padding)
    grid = np.zeros((height * y_dim + 1 + padding // 2, width *
                     x_dim + 1 + padding // 2, 3), dtype=np.uint8)
    k = 0
    for y in range(y_dim):
        for x in range(x_dim):
            if k >= img_num:
                break
            start_y = y * height + 1 + padding // 2
            end_y = start_y + height - padding
            start_x = x * width + 1 + padding // 2
            end_x = start_x + width - padding
            np.copyto(grid[start_y:end_y, start_x:end_x, :], datanp[k])
            k += 1
    imageio.imwrite(
        '{}/fake_samples_epoch_{}.png'.format(output_dir, epoch), grid)


def facc(label, pred):  # custom acc compare function
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


def train():
    image_pool = ImagePool(pool_size)
    metric = mx.metric.CustomMetric(facc)

    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)

    for epoch in range(start_epoch, epochs):
        tic = time.time()  # begin time of epoch
        btic = time.time()  # begin time of iterator

        iter = 0
        # for batch in train_data:
        for i, (real_in, real_out) in enumerate(train_data):
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            # real_in = batch.data[0].as_in_context(ctx)
            # real_out = batch.data[1].as_in_context(ctx)
            real_in = real_in.as_in_context(ctx)
            real_out = real_out.as_in_context(ctx)

            fake_out = netG(real_in)
            #fake_concat = image_pool.query(nd.concat(real_in, fake_out, dim=1))
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history images
                output = netD(fake_out)
                fake_label = nd.zeros(output.shape, ctx=ctx)
                errD_fake = GAN_loss(output, fake_label)
                metric.update([fake_label, ], [output, ])

                # Train with real image
                #real_concat = nd.concat(real_in, real_out, dim=1)
                output = netD(real_out)
                real_label = nd.ones(output.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD = (errD_real + errD_fake) * 0.5
                errD.backward()
                metric.update([real_label, ], [output, ])

            trainerD.step(real_in.shape[0])

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                fake_out = netG(real_in)

                #fake_concat = nd.concat(real_in, fake_out, dim=1)
                output = netD(fake_out)
                real_label = nd.ones(output.shape, ctx=ctx)
                errG = GAN_loss(output, real_label) + L1_loss(real_out, fake_out) * lambda1
                errG.backward()

            trainerG.step(real_in.shape[0])

            # Print log infomation every ten batches
            if iter % 40 == 0:
                name, acc = metric.get()
                logging.info(
                    'D loss = %f, G loss = %f, D acc = %f at iter %d epoch %d, %.2f samples/s'
                    % (nd.mean(errD).asscalar(), nd.mean(errG).asscalar(), acc, iter, epoch,
                       batch_size / (time.time() - btic)))
            iter = iter + 1
            btic = time.time()

        name, acc = metric.get()
        metric.reset()
        stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
        logging.info('%s D acc at epoch %d: %s=%f, time:%f ' % (stamp, epoch, name, acc, time.time() - tic))


        if epoch % 2 == 0 and epoch > 0:
            dump_param(epoch)
            test(epoch)


train()
#test(1000, 100)




