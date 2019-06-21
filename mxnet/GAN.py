from __future__ import print_function
import matplotlib as mpl
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
import numpy as np
from datetime import datetime
import os
import time

ctx = mx.gpu(0)
##############################
#generate some 'REAL' data

X = nd.random_normal(shape=(1000, 2))
A = nd.array([[1, 2], [-0.1, 0.5]])
b = nd.array([1, 2])
X = nd.dot(X, A) + b
Y = nd.ones(shape=(1000, 1))

# and stick them into an iterator
batch_size = 100
train_data = mx.io.NDArrayIter(X, Y, batch_size, shuffle=True)

##############################
# define network
# build the generator
netG = nn.Sequential()
with netG.name_scope():
    netG.add(nn.Dense(2))

# build the discriminator (with 5 and 3 hidden units respectively)
netD = nn.Sequential()
with netD.name_scope():
    netD.add(nn.Dense(5, activation='tanh'))
    netD.add(nn.Dense(3, activation='tanh'))
    netD.add(nn.Dense(2))

# loss
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# initialize the generator and the discriminator
netG.initialize(mx.init.Normal(0.02), ctx=ctx)
netD.initialize(mx.init.Normal(0.02), ctx=ctx)

# trainer for the generator and the discriminator
trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

##############################
# train the two networks
real_label = mx.nd.ones((batch_size,), ctx=ctx)
fake_label = mx.nd.zeros((batch_size,), ctx=ctx)
metric = mx.metric.Accuracy()

stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
for epoch in range(50):
    tic = time.time()
    train_data.reset()
    for i, batch in enumerate(train_data):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real_t
        data = batch.data[0].as_in_context(ctx)
        noise = nd.random_normal(shape=(batch_size, 2), ctx=ctx)

        with autograd.record():
            real_output = netD(data)
            errD_real = loss(real_output, real_label)

            fake = netG(noise)
            # 生成一个新匿名变量，等于fake从netG的计算图中断开后的值，backward的时候就不用做无用功，不计算netG的参数的梯度，虽然计算了也没事，因为trainerD只会更新netD的参数
            # 注意，fake本身并没有从netG中detach哈
            fake_output = netD(fake.detach() )
            errD_fake = loss(fake_output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()

        trainerD.step(batch_size)
        metric.update([real_label,], [real_output,])
        metric.update([fake_label,], [fake_output,])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            fake = netG(noise)  #加上这一步保险一点：如果上面那一步没有做detach产生新变量，执行起来逻辑也正确
            output = netD(fake)
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch_size) #只会更新G网络的参数，因为trainerG里只包含G网络的参数

    name, acc = metric.get() #准确率应该接近 50%是目标吧
    metric.reset()
    print('binary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    #print('time: %f' % (time.time() - tic))
    if (epoch+1) % 10 == 0 :
        noise = nd.random_normal(shape=(200, 2), ctx=ctx)
        fake = netG(noise)
        plt.scatter(X[:,0].asnumpy(), X[:,1].asnumpy())
        plt.scatter(fake[:,0].asnumpy(), fake[:,1].asnumpy())
        plt.show()

