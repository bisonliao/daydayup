"""
使用mxnet演示卷积神经网络

"""
import mxnet.gluon as gluon
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.ndarray as nd
import time
import numpy as np
from  load_data_from_file import MyFileDataset
import gluoncv as gcv
from gluoncv.utils import export_block



##################################################
# arguments
num_outputs = 10
#num_examples = 10000
epochs = 20
lr = 0.01
ctx = mx.gpu()
batch_size=100

seed = int(time.time())
mx.random.seed(seed)
mx.random.seed(seed, ctx)


##################################################
# training samples source
def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    return data, label

dataset = gluon.data.vision.CIFAR10('./data', train=True, transform=transformer)
train_example_num = len(dataset)
train_data = gluon.data.DataLoader(dataset,
    batch_size=batch_size, shuffle=True, last_batch='discard')

dataset = gluon.data.vision.CIFAR10('./data', train=False, transform=transformer)
test_example_num = len(dataset)
test_data = gluon.data.DataLoader(dataset,
    batch_size=batch_size, shuffle=False, last_batch='discard')

print("samples number:",train_example_num, " ", test_example_num)

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        num = data.shape[0]
        output = net(data)
        output = nd.softmax(output)
        predictions = output.argmax(axis=1)
        acc.update(preds=predictions, labels=label)
    em = acc.get()
    return em[1]


##################################################
# define network
def create_net():
    alex_net = gluon.nn.Sequential()
    with alex_net.name_scope():
        #  First convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=96, kernel_size=11, strides=(4,4), activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
        #  Second convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2,2)))
        # Third convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
        # Fourth convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
        # Fifth convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
        # Flatten and apply fullly connected layers
        alex_net.add(gluon.nn.Flatten())
        alex_net.add(gluon.nn.Dense(4096, activation="relu"))
        alex_net.add(gluon.nn.Dense(4096, activation="relu"))
        alex_net.add(gluon.nn.Dense(10))
    return alex_net

net = create_net()
net.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)

trainer = gluon.Trainer(net.collect_params(), "sgd", {"wd":0.00})
trainer.set_learning_rate(lr)
my_softmax_loss =gluon.loss.SoftmaxCrossEntropyLoss()


##################################################
# train the network
# 不需要每次梯度清 0，因为新梯度是写进去，而不是累加

for e in range(epochs):
    loss_sum = 0
    for i, (data, label) in enumerate(train_data):

        # data shape is [batch_size, input_size] : batch_size X input_size matrix
        # label shape is [batch_size, ]: (batch_size) scalars
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        with autograd.record():
            output = net(data) # the forward iteration
            loss = my_softmax_loss(output, label)

        # now , output shape is [batch_size,2]
        # loss shape is [batch_size,]
        loss.backward() # 可以传入一个同loss.grad形状的矩阵，作为梯度的数乘的系数，默认为1，所谓的头梯
        loss_sum += nd.mean(loss).asscalar()
        trainer.step(batch_size) # update the wb parameters

    loss_sum = loss_sum / (i+1)
    accuracy = evaluate_accuracy(test_data, net)
    print("epoch:", e, " loss:", loss_sum, " acc:", accuracy)


##################################################
# save net parameters and restore
net.save_parameters('e:/alexnet.params')

net2 = create_net()
net2.load_parameters('e:/alexnet.params', ctx=ctx)

##################################################
# take a batch from test data for visulization
for i, (data, label) in enumerate(test_data):
    if i < 10:
        continue
    pictures = data
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)
    output = net2(data)
    output = nd.argmax(output, axis=1)
    break
labeltxt = [
"airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"
]

print(output)

import matplotlib.pyplot as plt



def show_pic(data):
    if data.shape != (10, 3, 224, 224):
        print("shape mismatch")
        return
    pic = data.copy()
    for i in range(pic.shape[0]):
        onepic = pic[i]
        plt.imshow(onepic.asnumpy())
        plt.show()
    return
show_pic(pictures)








