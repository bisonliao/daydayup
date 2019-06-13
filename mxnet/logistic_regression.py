"""
使用mxnet的神经网络拟合logistic回归的例子
也可以直接用一层Dense经sigmoid作为网络结构，那就是真正的lr了
"""
import mxnet.gluon as gluon
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.ndarray as nd
import time
from  load_data_from_file import MyFileDataset
import gluoncv as gcv
from gluoncv.utils import export_block



##################################################
# arguments
num_inputs = 2
num_outputs = 2
num_examples = 200 #样本数太多（例如1万个）的话，第一次epoch就把模型训练好了，看不到随着epoch准确率明显的提升，所以我特意设置小一点
epochs = 100
lr = 0.1
ctx = mx.gpu()
batch_size=100

seed = 1 #int(time.time())
mx.random.seed(seed)
mx.random.seed(seed, ctx)


##################################################
# training samples source
def real_fn(X):
    z = 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2
    return 1 / (1+ nd.exp(-z))


def generate_data(data_num):
    X = nd.random_uniform(-10, 10,shape=(data_num, num_inputs),)
    #noise = 0.01 * nd.random_normal(shape=(num_examples,))
    y = nd.round(real_fn(X))
    # load data from memory
    return gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),batch_size=batch_size, shuffle=True)

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

train_data = generate_data(num_examples)
test_data = generate_data(200)

##################################################
# define network
# actually, for this kind of simple data, only the
# last Dense layer is needed
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(256, activation="relu"))# 1st layer (256 nodes)
    net.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
    namedLayer = gluon.nn.Dense(num_outputs, activation="sigmoid") # give the layer a name , for access convenient
    namedLayer.weight.lr_mult = 2.0
    net.add(namedLayer)
net.initialize(ctx=ctx,)
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate":0.1,"wd":0.0001}) #accept learning_rate, wd (weight decay), clip_gradient, and lr_scheduler.
trainer.set_learning_rate(lr) #也就这个函数，没有set_wd哈，其他参数也没有函数可以改
my_softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()


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
        loss.backward()
        loss_sum += nd.mean(loss).asscalar()
        trainer.step(batch_size) # update the wb parameters

    loss_sum = loss_sum / (i+1)
    accuracy = evaluate_accuracy(test_data, net)
    print("epoch:" ,e, " loss:", loss_sum, " acc:", accuracy)



##################################################
# show how to access the parameters and gradients
print(namedLayer.weight.shape)
print(namedLayer.weight.data(ctx=ctx)[0,1])
print(namedLayer.weight.grad(ctx=ctx)[0,1])
print(namedLayer.bias.grad(ctx=ctx)[0])

##################################################
# save net parameters
net.save_parameters('e:/logistic.params')
#export_block('e:/lr.json',net) #this does NOT work! export_block is from gluoncv


##################################################
# restore the trained net and use it
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(256, activation="relu"))# 1st layer (256 nodes)
    net2.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
    net2.add(gluon.nn.Dense(2, activation="sigmoid"))


net2.load_parameters('e:/logistic.params', ctx=ctx)

test_size = 100
X = nd.random_uniform(-100, 100, shape=(test_size, num_inputs), ctx=ctx)
Y = (real_fn(X)+0.5).astype("int32").asnumpy()
YY = net2(X)
YY = nd.argmax(YY, axis=1)


acc = 0
for i in range(test_size):
    if Y[i] == YY[i]:
        acc=acc+1
print("accuracy:", acc/test_size)
#print(Y)
#print(YY)




