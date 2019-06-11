"""
使用mxnet实现线性回归的例子
"""
import mxnet.gluon as gluon
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.ndarray as nd
from  load_data_from_file import MyFileDataset
import gluoncv as gcv
from gluoncv.utils import export_block



##################################################
# arguments
num_inputs = 2
num_outputs = 1
num_examples = 10000
epochs = 50
lr = 0.01
ctx = mx.gpu()
batch_size=100

mx.random.seed(2019)
mx.random.seed(2019, ctx)


##################################################
# training samples source
def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


def generate_data():
    X = nd.random_normal(shape=(num_examples, num_inputs))
    noise = 0.01 * nd.random_normal(shape=(num_examples,))
    y = real_fn(X) + noise
    # load data from memory
    #return gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),batch_size=batch_size, shuffle=True)
    #load data from file or other source
    return gluon.data.DataLoader(MyFileDataset(), batch_size=batch_size, shuffle=True)


train_data = generate_data()

##################################################
# define network
# actually, for this kind of simple data, only the
# last Dense layer is needed
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(256, activation="relu"))# 1st layer (256 nodes)
    net.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
    namedLayer = gluon.nn.Dense(1) # give the layer a name , for access convenient
    net.add(namedLayer)
net.initialize(ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), "sgd")
trainer.set_learning_rate(lr)
my_square_loss = gluon.loss.L2Loss()


##################################################
# train the network
# question: when and who clean the gradient?
for e in range(epochs):
    loss_sum = 0
    for i, (data, label) in enumerate(train_data):
        # data shape is [batch_size, input_size] : batch_size X input_size matrix
        # label shape is [batch_size, ]: (batch_size) scalars
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data) # the forward iteration
            loss = my_square_loss(output, label)
        # now , output shape is [batch_size,1]
        # loss shape is [batch_size,]
        loss.backward()
        loss_sum += nd.sum(loss).asscalar() / loss.shape[0]
        trainer.step(batch_size) # update the wb parameters
    loss_sum = loss_sum / (i+1)
    print("epoch["+str(e)+"], loss:"+str(loss_sum))

##################################################
# show how to access the parameters and gradients
print(namedLayer.weight.shape)
print(namedLayer.weight.data(ctx=ctx)[0,1])
print(namedLayer.weight.grad(ctx=ctx)[0,1])
print(namedLayer.bias.grad(ctx=ctx)[0])

##################################################
# save net parameters
net.save_parameters('e:/lr.params')
#export_block('e:/lr.json',net) #this does NOT work! export_block is from gluoncv


##################################################
# restore the trained net and use it
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(256, activation="relu"))# 1st layer (256 nodes)
    net2.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
    net2.add(gluon.nn.Dense(1)) # give the layer a name , for access convenient

net2.load_parameters('e:/lr.params', ctx=ctx)
X = nd.random_normal(shape=(10, num_inputs), ctx=ctx)
Y = real_fn(X)
YY = net2(X)

print(Y)
print(YY.T[0])
print(nd.subtract(YY.T[0] , Y) )


