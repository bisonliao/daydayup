'''
孪生网络，用来判断两张手写数字是否是同一个数
也可以为一张图片生成摘要
'''

import mxnet.gluon as gluon
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.ndarray as nd
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random

batch_size = 128
context = mx.gpu(0)
lr = 0.01
epochs = 10

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

def show(data):
    onepic = data.reshape(28,28,1)
    onepic = nd.tile(onepic, (1, 1, 3))  # 扩为 (28, 28, 3) BGR图片
    plt.imshow(onepic.asnumpy())
    plt.show()

def gene_train_examples():
    left = []
    right = []
    label = []
    pos_num = 0
    neg_num = 0
    siamese_data=[]
    dataset1 = mx.gluon.data.vision.MNIST(train=True, transform=transform)
    dataset2 = mx.gluon.data.vision.MNIST(train=True, transform=transform)
    file_index = 1
    for it1, (data1, label1) in enumerate(dataset1):
        data1 = data1.reshape((1,28,28))

        for it2, (data2, label2) in enumerate(dataset2):
            data2 = data2.reshape((1, 28, 28))

            if it1 == it2:
                continue
            if label1 == label2:
                if pos_num > 2 * neg_num:
                    continue

                left.append(data1)
                right.append(data2)
                label.append(1.0)
                pos_num += 1
            else:
                if neg_num > 2 * pos_num:
                    continue
                left.append(data1)
                right.append(data2)
                label.append(0.0)
                neg_num += 1

            if (pos_num + neg_num) >= batch_size: # get a whole batch
                left_nd = nd.zeros(shape = (batch_size, 1, 28, 28))
                for i in range(batch_size):
                    left_nd[i] = left[i]


                right_nd = nd.zeros(shape =(batch_size, 1, 28, 28))
                for i in range(batch_size):
                    right_nd[i] = right[i]

                label_nd = nd.array(label)

                siamese_data.append((left_nd, right_nd, label_nd))

                left = []
                right = []
                label = []
                pos_num = 0
                neg_num = 0
                break

        if len(siamese_data) >= 1000: # get a whole data set
            path="./data/siamese_data_{}.pickle".format(file_index)
            file_index +=1
            random.shuffle(siamese_data)
            with open(path, "wb") as f:
                pickle.dump(siamese_data, f)
            siamese_data = []
        if file_index > 2 :
            break;




#gene_train_examples()
#exit(0)

train_data=[]
test_data = []
with open("./data/siamese_data_1.pickle", "rb") as f:
    train_data = pickle.load(f)
with open("./data/siamese_data_2.pickle", "rb") as f:
    test_data = pickle.load(f)

##################################################
# define network
def create_net():
    alex_net = gluon.nn.Sequential()
    with alex_net.name_scope():
        #  First convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=96, kernel_size=5,  activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=2, strides=(2,2)))
        #  Second convolutional layer
        alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5,activation='relu'))
        alex_net.add(gluon.nn.MaxPool2D(pool_size=2, strides=(2,2)))

        alex_net.add(gluon.nn.Flatten())
        alex_net.add(gluon.nn.Dense(500, activation="relu"))
        alex_net.add(gluon.nn.Dense(10))
    return alex_net

net = create_net()
net.initialize(mx.init.Xavier(magnitude=2.24),ctx=context)

trainer = gluon.Trainer(net.collect_params(), "sgd", {"wd":0.00})
trainer.set_learning_rate(lr)

# according to standard loss function such as SoftmaxCrossEntropyLoss/L2Loss
# loss value shape should be (batch_size, )
def my_softmax_loss(l_o, r_o, y):
    margin = 1.0
    d = nd.norm(l_o - r_o, axis=1) #
    dd = nd.relu(margin -d )
    l = y*d*d+(1-y)*dd*dd
    return l

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    margin = 0.5
    for i, (left, right, label) in enumerate(data_iterator):
        left = left.as_in_context(context)
        right = right.as_in_context(context)
        label = label.as_in_context(context)

        l_o = net(left)
        r_o = net(right)
        d = nd.norm(l_o - r_o, axis=1).asnumpy()
        d = [ (1 if d[i] < margin else 0) for i in range(len(l_o))]
        predictions = nd.array(d)
        acc.update(preds=predictions, labels=label)
        if i > 10:
            break
    em = acc.get()
    return em[1]

##################################################
# train the network
# 不需要每次梯度清 0，因为新梯度是写进去，而不是累加

for e in range(epochs):
    loss_sum = 0
    for i, (left,right, label) in enumerate(train_data):


        left = left.as_in_context(context)
        right = right.as_in_context(context)
        label = label.as_in_context(context)

        with autograd.record():
            l_o = net(left) # the forward iteration
            r_o = net(right)
            loss = my_softmax_loss(l_o, r_o, label)
        loss.backward()
        #loss.backward()
        loss_sum += nd.mean(loss).asscalar()
        trainer.step(batch_size) # update the wb parameters

    loss_sum = loss_sum / (i+1)
    accuracy = evaluate_accuracy(test_data, net)
    print("epoch:", e, " loss:", loss_sum, " acc:", accuracy)


##################################################
# save net parameters and restore
net.save_parameters('e:/mnist_siamese.params')
