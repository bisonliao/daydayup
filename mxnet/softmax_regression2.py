"""
使用mxnet演示多分类回归，且加入了dropout层，dropout层通常用来防止过拟合
自定义了一个继承自gluon.Block的类，实现了网络的定义，有了更好的封装
在其forward函数里可以方便的检查每一层数据传递的情况
相比只使用一个全连接层，这里使用了更深的网络，mnist识别准确率明显更高
This model is sometimes called multiclass logistic regression. Other common names for it include softmax regression and multinomial regression.

http://zh.gluon.ai/index.html
https://gluon.mxnet.io/

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
num_inputs = (28*28)
num_outputs = 10
#num_examples = 10000
epochs = 50
lr = 0.01
ctx = mx.gpu()
batch_size=100

seed = int(time.time())
mx.random.seed(seed)
mx.random.seed(seed, ctx)


##################################################
# training samples source
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

dataset = mx.gluon.data.vision.MNIST(train=True, transform=transform)
train_example_num = len(dataset)
train_data = mx.gluon.data.DataLoader(dataset, batch_size, shuffle=True)

dataset = mx.gluon.data.vision.MNIST(train=False, transform=transform)
test_example_num = len(dataset)
test_data = mx.gluon.data.DataLoader(dataset,10, shuffle=False)
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
class MLP(gluon.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(64)
            self.dense1 = gluon.nn.Dense(64)
            self.drop0 = gluon.nn.Dropout(0.5)
            self.dense2 = gluon.nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        x = nd.relu(self.dense1(x))
        x = self.drop0(x)
        x = self.dense2(x)
        return x

net = MLP()
net.initialize(ctx=ctx)

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
net.save_parameters('e:/softmax.params')

net2 = MLP()
net2.load_parameters('e:/softmax.params', ctx=ctx)

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

print(output)

import matplotlib.pyplot as plt

#显示图片。不是很好理解，自己写的话就啰嗦一点慢慢转。这部分代码不重要
'''
im = nd.transpose(pictures,(1,0,2,3))
im = nd.reshape(im,(28,10*28,1)) # 搞成一张大图，28行高，28X10列宽，每个像素1个数
imtiles = nd.tile(im, (1,1,3))
plt.imshow(imtiles.asnumpy())
plt.show()
'''
#自己搞个函数
def show_pic(data):
    if data.shape != (10, 28,28,1):
        print("shape mismatch")
        return
    pic = data.copy()
    for i in range(pic.shape[0]):
        onepic = pic[i]
        onepic = nd.tile(onepic, (1,1,3)) #扩为 (28, 28, 3) BGR图片
        plt.imshow(onepic.asnumpy())
        plt.show()
    return
show_pic(pictures)







