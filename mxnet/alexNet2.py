"""
使用mxnet演示卷积神经网络AlexNet

观察到原来代码运行的时候，GPU使用率很低，对train data数据的加载做了修改，
但对速度提升没有帮助，GPU占用率还是很低

我感觉主要是因为图片上采样使用CPU来做的，且没有并发
由于内存的限制，上采样每个epoch都重复做，浪费


"""
import mxnet.gluon as gluon
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.ndarray as nd
import time
import numpy as np
import mxnet.gluon.data as data
import random
from  load_data_from_file import MyFileDataset
import gluoncv as gcv
from gluoncv.utils import export_block



##################################################
# arguments
num_outputs = 10
#num_examples = 10000
epochs = 20
lr = 0.01
ctx = mx.gpu(0)
batch_size=200

seed = int(time.time())
mx.random.seed(seed)
mx.random.seed(seed, ctx)


##################################################
# training samples source
def transform(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    return data, label

# create a data loader by myself, hope  to acceleate data loading,
# but actually it is not helpful,even slower than standard Loader
class MyLoader(data.Dataset):
    def __init__(self, datasource,batchsz):
        self.source = datasource
        self.index = 0
        self.len = len(datasource)
        (d,l) = datasource[0]
        (d,l) = transform(d,l)
        self.img_shape = d.shape
        self.batchsz = batchsz
        return

    def __getitem__(self, item):
        if item >= (self.len // self.batchsz):
            raise StopIteration
        (c,h,w) = self.img_shape

        data = nd.zeros(shape=(self.batchsz,c,h,w),ctx=mx.gpu())
        label = nd.zeros(shape=(self.batchsz,),ctx=mx.gpu())
        for i in range(self.batchsz):
            (dv,lv) = self.source[self.index]
            (dv, lv) = transform(dv, lv)
            data[i] = dv
            label[i] = lv
            self.index = (self.index+1)%self.len
        return (data,label)
        #return self.source[item]
    def __len__(self):
        return len(self.source) // self.batchsz

# load train samples in memory to accelerate the speed
# actually it is not helpful
def create_memdb():
    dataset = gluon.data.vision.CIFAR10('./data', train=True)
    train_example_num = len(dataset)
    print("train samples number:", train_example_num)
    memset = [dataset[i] for i in range(train_example_num) ]
    random.shuffle(memset)
    return MyLoader(memset,batch_size)

train_data = create_memdb()

#use ordinary Data Loader
dataset = gluon.data.vision.CIFAR10('./data', train=False, transform=transform)
test_example_num = len(dataset)
test_data = gluon.data.DataLoader(dataset,
        batch_size=10, shuffle=False, last_batch='discard',pin_memory=True)

print("test samples number:", test_example_num)

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
        if i > 20:
            break
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
batchnr = len(train_data)
for e in range(epochs):
    loss_sum = 0
    loss_cnt = 0
    for i, (data,label) in enumerate(train_data):

        with autograd.record():
            output = net(data) # the forward iteration
            loss = my_softmax_loss(output, label)

        loss.backward()
        loss_sum += nd.mean(loss).asscalar()
        loss_cnt += 1
        trainer.step(batch_size)  # update the wb parameters
        if (i % (  batchnr //3)) ==0 and i != 0\
                or i==(batchnr-1):
            loss_sum = loss_sum / (loss_cnt + 1)
            print("loss:", loss_sum)
            loss_sum = 0
            loss_cnt = 0
    accuracy = evaluate_accuracy(test_data, net)
    print("epoch:", e," acc:", accuracy)





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
output = output.astype("int32").asnumpy()
for i in range(len(output)):
    index =  output[i]
    print(labeltxt[index])


import matplotlib.pyplot as plt

def chw2hwc(data):
    c=data.shape[0]
    h=data.shape[1]
    w=data.shape[2]
    ch1 = nd.flatten(data[0,:,:]).asnumpy()
    if c == 3 :
        ch2 = nd.flatten(data[1,:,:]).asnumpy()
        ch3 = nd.flatten(data[2, :, :]).asnumpy()
    else:
        ch2 = nd.flatten(data[0, :, :]).asnumpy()
        ch3 = nd.flatten(data[0, :, :]).asnumpy()

    data = nd.array([ch1,ch2,ch3]).T
    data = nd.flatten(data)
    data = data.reshape((h,w,c))
    data = mx.image.imresize(data, 32, 32) / 255.0
    return data


def show_pic(data):
    if data.shape != (10, 3, 224, 224):
        print("shape mismatch")
        return
    pic = data.copy()
    for i in range(pic.shape[0]):
        onepic = pic[i]
        onepic = chw2hwc(onepic)
        plt.imshow(onepic.asnumpy())
        plt.show()
    return
show_pic(pictures)








