"""
使用mxnet演示logistic回归
直接用一层Dense经sigmoid作为网络结构
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
num_outputs = 1
num_examples = 200 #样本数太多（例如1万个）的话，第一次epoch就把模型训练好了，看不到随着epoch准确率明显的提升，所以我特意设置小一点
epochs = 100
lr = 0.01
ctx = mx.gpu()
batch_size=100

seed = int(time.time())
mx.random.seed(seed)
mx.random.seed(seed, ctx)


##################################################
# training samples source
def real_fn(X):
    z = 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2
    return 1 / (1+ nd.exp(-z))


def generate_data(data_num):
    # X的取值范围很重要，如果绝对值比较大，那么wX+b的值就大，经过sigmoid函数的导数的时候，导致梯度消失，loss值下降很慢
    # 如果X的取值范围确实很大，要做标准化处理
    # 而且学习到的系数也不等于real_fn里的系数2，-3.4,4.2
    X = nd.random_uniform(-1, 1,shape=(data_num, num_inputs),)
    #标签 y转为1,0这样的数值后，再转为float32这一步很重要，否则后面loss的计算会报错
    # ndarray之间的相互运算，例如两个nd相加，他们的dtype要求一致，
    # 否则报错，类似这样的： Incompatible attr in node  at 1-th input: expected int32, got float32
    y = (real_fn(X)+0.5).astype("int32").astype("float32")
    # load data from memory
    return gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),batch_size=batch_size, shuffle=True)

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        num = data.shape[0]
        output = net(data)
        predictions = (output+0.5).astype("int32").reshape((num,))
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

train_data = generate_data(num_examples)
test_data = generate_data(200)

##################################################
# define network
# actually, for this kind of simple data, only the
# last Dense layer is needed
net = gluon.nn.Dense(1,activation="sigmoid")
net.initialize(ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), "sgd")
trainer.set_learning_rate(lr)

# according to standard loss function such as SoftmaxCrossEntropyLoss/L2Loss
# loss value shape should be (batch_size, )
def my_softmax_loss(output, y):
    # two implements, I think sencond one is more standard.
    if False:
        # returning a NDArray with shape(1,) is ok
        loss_value = -nd.nansum(  y * nd.log(output) + (1-y) * nd.log(1-output)   ) / output.shape[0]
        return loss_value
    else:
        # returning a NDArray with shape(batch_size,)
        y = y.reshape(-1,1)
        loss_value = -( y * nd.log(output) + (1-y) * nd.log(1-output)  )
        return  loss_value.reshape(-1,)




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
    print("epoch:", e, " loss:", loss_sum, " acc:", accuracy)
    if (e % 5) == 0:
        w1 = net.weight.data(ctx=ctx)[0, 0].asscalar()
        w2 = net.weight.data(ctx=ctx)[0, 1].asscalar()
        b = net.bias.data(ctx=ctx)[0].asscalar()
        print("!!!", w1, " ", w2, " ", b)


##################################################
# save net parameters and restore
net.save_parameters('e:/logistic2.params')

net2 = gluon.nn.Dense(1, activation="sigmoid")
net2.load_parameters('e:/logistic2.params',ctx = ctx)
val_data = generate_data(1000)
print("val date acc:", evaluate_accuracy(val_data, net2))





