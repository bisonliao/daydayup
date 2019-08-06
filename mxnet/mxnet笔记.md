### 1、多批次的梯度累加

mxnet gluon不需要像pytorch caffe那样手动清零梯度，因为新梯度默认是写进去，而不是累加。

对于小显存显卡的小批量数据的训练，可以多次批量累加梯度，模拟出大显存大批量的效果：

```python
#默认是write，改为累加
for p in net.collect_params().values():
    p.grad_req="add"
# ... 
# 在每次trainer.step更新参数后，将梯度清零。step的batchsz参数要做适当的修改
        if (i%2)==0 and i != 0:
            trainer.step(batch_size*2) # update the wb parameters
            net.collect_params().zero_grad()
```

![](img/grad_req.jpg)

可以这样来访问梯度和参数值等数据：

```python
print(namedLayer.weight.shape)
print(namedLayer.weight.data(ctx=ctx)[0,1])
print(namedLayer.weight.grad(ctx=ctx)[0,1])
print(namedLayer.bias.grad(ctx=ctx)[0])
```

默认情况下不需要手动清理梯度，但mxnet的网络的参数需要手动调用initialize()函数来初始化，或者load_parameters()从参数文件中加载。

### 2、自动求解梯度

自动求解梯度比较重要，所以基本上是李沐的教程中那一章的全部记录。

#### 2.1 基本理解

with autograd.record()会跟踪代码块范围内NDArray的相关操作，每个操作都有对应的反向传播函数，例如下面的代码

1. mean()函数对应的反向传播梯度是1/N（N是元素个数），
2. max()函数的梯度，类似max pooling，会有mask记录哪个元素被取值，对应的梯度为1，其他为0：

```python
x = nd.array([[1,2,3],[3,2,1]])
x.attach_grad()
with autograd.record(False):
    y =  nd.mean(x)
    y.backward()

print("y:", y)
print("grad:", x.grad)
```

输出如下：

```
y: 
[2.]
<NDArray 1 @cpu(0)>
grad: 
[[0.16666667 0.16666667 0.16666667]
 [0.16666667 0.16666667 0.16666667]]
<NDArray 2x3 @cpu(0)>
```

```python
x = nd.array([[1,2,3],[3,2,1]])
x.attach_grad()
with autograd.record(False):
    y =  nd.max(x)
    y.backward()

print("y:", y)
print("grad:", x.grad)
```

输出如下：

```
y: 
[3.]
<NDArray 1 @cpu(0)>
grad: 
[[0. 0. 1.]
 [1. 0. 0.]]
<NDArray 2x3 @cpu(0)>
```

有一点不好理解的是下面的代码：

```python
x = nd.array([[1,2,3],[3,2,1]])
x.attach_grad()
with autograd.record(False):
    y =  nd.mean(x) + 2 * x
    y.backward()

print("y:", y)
print("grad:", x.grad) #这里显示梯度为3，而不是我们理解的2+1/6

# 根据梯度的定义，自变量加上一个小小的delta，看看因变量变化多少
# 发现确实变化了3倍，而不是2.666倍
x = x + 0.00001
yy = nd.mean(x) + 2 * x
print(yy - y)

# 但这样又怎么解释呢？
x = nd.array([[1,2,3],[3,2,1]])
x = x + nd.array([[0.00001, 0, 0],[0,0,0]])
yy = nd.mean(x) + 2 * x
print(yy - y)
```

后来理解了：

反向传播都是从一个标量开始求导，如果正向传播最后得出的是一个向量或者矩阵，那么mxnet会先把这里面的元素都加起来得到标量，再开始反向传播。

有了这个规则，上面的梯度为3而不是2+1/6 就好解释了：

```python
  y= [ [x1*2 + (x1+x2+...+x6)/6,  x2*2 + (x1+x2+...+x6)/6 , x3*2 + (x1+x2+...+x6)/6 ],
	   [x1*2 + (x1+x2+...+x6)/6,  x2*2 + (x1+x2+...+x6)/6 , x3*2 + (x1+x2+...+x6)/6 ] ]
  #backward的时候，先把y的个元素加起来，得到标量，再对X的个元素求偏导：
  loss=x1*2 + x2*2 +...+ x6*2 + 6*( (x1+x2+...+x6)/6 )
  loss=x1*3 + x2*3 +...+ x6*3
```

引自李沐的文档：

```
So MXNet will sum the elements in y to get the new variable by default, and then find the analytical gradient of the variable with respect to x evaluated at its current value y/dx 
```

#### 2.2 detach和attach_grad的运用

如果一个变量（通常是NDArray类型的变量）执行detach()，就会返回一个“常量”NDArray，常量失去了mxnet记忆的compute graph，mxnet不记得该常量从何而来，链式求导过程遇到该常量就会结束，不再继续传递，但该常量本身是会有梯度的。

下面的代码，u相当于一个常量作用在x上，所以z对x的导数就等于u的值：

```python
x = nd.array([[1,2,3],[3,2,1]])
x.attach_grad()
with autograd.record():
	y = x * x
	u = y.detach()
	z = u * x
z.backward()
print(x.grad) #输出u的值，也就是y的值
y.backward()
print(x.grad) #输出2X
```

输出如下：

```
[[1. 4. 9.]
 [9. 4. 1.]]
<NDArray 2x3 @cpu(0)>

[[2. 4. 6.]
 [6. 4. 2.]]
<NDArray 2x3 @cpu(0)>
```

对一个变量x执行attach_grad()，相当于调用了x=x.detach()，当然不只是做了detach，还有分配梯度内存等操作。链式求导到该变量截止，该变量相当于一个常量。例如下面代码，x和u的梯度都是1，y的梯度则为0：

```python
x=nd.ones(4)
y = nd.ones(4) * 2
x.attach_grad() #这里不能写成x=x.detach(),因为attach_grad() more than detach()
y.attach_grad()
with autograd.record():
    u = x * y
    u.attach_grad() # implicitly run u = u.detach()， 何必写成这个死样子呢？
    z = u + x
z.backward()
print(x.grad, u.grad, y.grad)
```

输出为：

```
[1. 1. 1. 1.]
<NDArray 4 @cpu(0)> 
[1. 1. 1. 1.]
<NDArray 4 @cpu(0)> 
[0. 0. 0. 0.]
<NDArray 4 @cpu(0)>
```

下面这段代码有点意思，z是y的函数，y是x的函数。为了逐层计算梯度（链式法则），y必须截留梯度信息：

```python
x=nd.ones(4)
x.attach_grad()
with autograd.record():
    y = 2*x
    y.backward() #计算y 对x 的梯度
    # y想截留前面z对自己的梯度
    y.attach_grad() # y：我想分配内存保存前面z对我的梯度，并且这个梯度不再向后传播了，因为隐含的调用了y=y.detach()
    z = y * y
	z.backward() # 计算z对y的梯度
print(x.grad,y.grad)
```

输出是：

```
[2. 2. 2. 2.]
<NDArray 4 @cpu(0)> 
[4. 4. 4. 4.]
<NDArray 4 @cpu(0)>
```

而这样写代码是得不到z对y的梯度的，y.grad为None：

```python
x=nd.ones(4)
x.attach_grad()
with autograd.record():
    y = 2*x
    z = y * y
z.backward()
print(x.grad,y.grad)
```

#### 2.3 头梯的运用

还是上面的例子，z是y的函数，y是x的函数

```pyton
x=nd.ones(4)
x.attach_grad()
with autograd.record():
    y = 2*x
    yy = y.detach() #yy用于截胡z对y的梯度
    yy.attach_grad()
    z=yy*yy
z.backward()
print(x.grad,yy.grad) #x的梯度为0，因为被yy截胡了
y.backward(yy.grad)# 计算z对x的梯度：利用z对y的梯度作为头梯系数乘一下y对x的梯度即可
print(x.grad)
```

输出为：

```
[0. 0. 0. 0.]
<NDArray 4 @cpu(0)> 
[4. 4. 4. 4.]
<NDArray 4 @cpu(0)>
[8. 8. 8. 8.]
<NDArray 4 @cpu(0)>
```

#### 2.4 python本身流程控制语句的计算图的追踪

即使前向传递过程中包含有if for等python流程控制语句，mxnet也能够追踪计算图和梯度关系，这个跟MaxPool里用位图记录前向传递使用了哪个值类似，autograd应该是有辅助变量来记录相关信息的。

```python
def f(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c

x=nd.ones(4)
x.attach_grad()
with autograd.record():
    y = f(x)
y.backward()
print(x.grad)
```

输出：

```
[512. 512. 512. 512.]
<NDArray 4 @cpu(0)>
```



### 3、自定义的backward

大多数时候，我们前向传播用NDArray已经定义好的函数和运算符来完成，不需要我们关注和实现反向传播，mxnet自动帮我们搞定。主要在以下场景需要注意尽量用NDArray成员函数：

1. 继承自Block的自定义的网络类的forward函数里
2. 继承自Block的自定义层的forward函数里
3. 自己定义的loss函数
4. with autograd.record() 的代码块里

例如我在实现孪生网络的loss函数时候，要尽量用NDArray的已有成员函数来完成，如果用一些循环语句对各个元素逐个操作（例如求解两个hash之间的距离d），那么autograd就不知道该怎么对该loss函数做反向传播了：

```python
def my_softmax_loss(l_o, r_o, y):
    margin = 1.0
    d = nd.norm(l_o - r_o, axis=1) #
    dd = nd.relu(margin -d )
    l = y*d*d+(1-y)*dd*dd
    return l
```

有时候前向传播计算太过奇特，无法用NDArray成员函数来组合，那就要自己实现backward函数了。

方法是自定义一个继承自mxnet.autograd.Function的子类，实现forward和backward函数。

以自定义sigmoid为例：

```python
class sigmoid(mx.autograd.Function):
    def forward(self, x):
        y = 1 / (1 + mx.nd.exp(-x))
        self.save_for_backward(y)
        return y

    def backward(self, dy):
        # backward takes as many inputs as forward's return value,
        # and returns as many NDArrays as forward's arguments.
        y, = self.saved_tensors
        return dy * y * (1-y)
    
func = sigmoid()
x = mx.nd.random.uniform(shape=(10,))
x.attach_grad()

with mx.autograd.record():
    m = func(x)
    m.backward()
dx = x.grad.asnumpy()
```

详细见：

```
https://beta.mxnet.io/api/gluon-related/_autogen/mxnet.autograd.Function.html
```

我一开始以为在继承自Block或者Loss类的自定义子类中实现backward方法能够达到这个目的，实际上不可以！

如果要修改某个自定义层（Block的子类）的backward计算过程，例如想实现自定义的sigmoid正反双向传播，直接使用上面自定义的sigmoid类即可：

在这个自定义层的forward函数中调用sigmoid的\_\_call\_\_即可，autograd在反向传播的时候会自动调用到sigmiod类的backward函数，类似这样的：

```python
class MyLayer(block.Block):
    def __init__(self,  **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.func = sigmoid()

    def forward(self, x):
        print("forward is called")
        return self.func(x)
```

似乎gluon接口里只有上面一种方式来实现，非gluon接口似乎还可以使用什么自定义的Op子类，这个我没有研究。



### 4、学习率、学习率倍数等超参数

可以这样来修改某参数的学习率倍数，默认是1.0：

```python
namedLayer.weight.lr_mult = 2.0
```

![](img/lr_mult.jpg)

Trainer（感觉对应caffe的solver）初始化的时候可以指定很多超参数：

```python
#第三个字典类型的参数可以设置：
#learning_rate, wd (weight decay), clip_gradient, and lr_scheduler.
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate":0.1,"wd":0.0001})
#单独为lr有个函数，其他超参数没有这个待遇，例如没有set_wd哈
trainer.set_learning_rate(lr) 
```

网络在初始化函数里可以指定整个网络的参数随机初始方法，默认为均匀分布。

各层在创建的时候也可以指定自己的初始化方法，包括以下参数：

1. output number
2. 激活函数
3. 参数初始化方法
4. 数值类型等等

以全连接层和卷积层为例：

![](img/layer_param.jpg)

详细见API文档：

```
https://mxnet.incubator.apache.org/api/python/index.html
```

### 5、自定义的数据读取方式

假设训练数据是来自我们自己形态比较特殊的数据源，例如log文件。那怎么做呢？

1. 自定义类继承自 mxnet.gluon.data.Dataset
2. 实现\_\_getitem\_\_ 和\_\_len\_\_ 函数
3. 创建该自定义类的实例，用做gluon.data.DataLoader实例的参数

下面是一个简单例子：

```python
import mxnet.gluon.data as data
import mxnet.ndarray as nd

num_examples = 1000
num_inputs = 2

X = nd.random_normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random_normal(shape=(num_examples,))
y = 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2 + noise
XY = data.ArrayDataset(X, y)

class MyFileDataset(data.Dataset):
    def __init__(self):
        return

    def __getitem__(self, item):
        return XY[item]
    def __len__(self):
        return num_examples
```

然后：

```pyton
train_data = gluon.data.DataLoader(MyFileDataset(), batch_size=batch_size, shuffle=True)
```

DataLoader也很方便自己实现，只要是满足python的可迭代的对象即可，兼容训练的时候通过enumerate()操作访问data loader中的数据。windows下mxnet提供的DataLoader不支持num_workers=cpu_nr参数，会抛异常，这样访问数据的效率就比较低了，GPU跑不满。

详细文档可见：

```
https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/datasets.html
```



### 6、自定义layer

gluon.Block是mxnet里非常重要的类，重要性与NDArray不相上下。

自定义layer就是写一个继承Block的子类，具体做法：

```
https://gluon.mxnet.io/chapter03_deep-neural-networks/custom-layer.html#Defining-a-(toy)-custom-layer
```

Block类非常重要的属性有：

1. params：类型为mxnet.gluon.parameter.ParameterDict，用于保存该block自身的参数，不保存子块的参数。这里面的参数在save_parameters/load_parameters的时候会被保存和装载。它主要有两个方法：
   1. get_constant()，创建/获取一个"常量"NDArray，该参数在训练过程中不需要梯度更新的
   2. get()，创建 /或者类似w b这样的NDArray参数，该参数在训练过程中不需要梯度更新的
2. _children：类型为OrderedDict，用于保存该Block的子块。\_children字典里保存的子块，会在该Block的initialize/load_parameters/save_parameters/\_\_call\_\_/collect_params等操作中自动照顾到。几种情况的变量会进入到该字典：
   1. self.xxx这样的Block属性，例如self.fc0=Dense(256)也会自动进入\_children字典。
   2. register_child()注册的变量，会进入\_children字典。
   3. Block的子类nn.Sequential的add()成员函数增加的Block，例如net.add(Dense(256))，该变量自动进入\_children字典。

示例代码：

```python
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.weight = self.params.get('weight', 
                                      init=mxnet.init.Xavier(magnitude=2.24),
                                      shape=(10, 15))
        self.c = self.params.get_constant( 'cons_time', 
                                           nd.random.uniform(shape=(20, 20)))
		print([(k, self.params[k]) for k in self.params.keys()])
        self.fc0 = nn.Dense(256)
        print([(k,self._children[k] ) for k in self._children.keys()])
mlp = MLP()
print(dir(mxnet.gluon.parameter)) #可以查看该module下有哪些类
help(mxnet.gluon.parameter.ParameterDict)#可以打印该类的文档信息
```

### 7、Train mode and predict mode

数据在net中前向传播的时候，有两种模式，train mode和predict模式，mxnet默认是predict模式。例如这段代码就是predict模式，打印显示false：

```python
ctx=mx.gpu(0)
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
net.load_parameters('e:/softmax.params', ctx=ctx)
inputdata = nd.ones(shape=(64,784),ctx=ctx)
net(inputdata)
print(autograd.is_training())
```

autograd.record()函数的第一个参数就是指定是否是training_mode，默认值为True。

模式会影响Dropout层，当predict_mode时，Dropout层不工作，可以用下面的代码验证：

```python
net = MLP()
net.initialize(ctx=ctx)

inputdata = nd.array([[1,2,3,4]],ctx=ctx)
with autograd.train_mode():
    output = net(inputdata)
    print(output)
    output = net(inputdata)
    print(output)
exit(0)
```

当train mode的时候，由于dropout随机丢弃一些节点值，所以两次output的值不一样；如果是predict mode，两次output的值就是一样的。

在Caffe里，Dropout层的逻辑也是这样的，非Train模式不工作：

```c++
if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
```

### 8、获取中间层的输出

我们有时候需要获得中间层的输出，例如查看卷积网络的feature map。mxnet下怎么做呢？

只能是逐层的前向传播，遇到目标层的输出就保存起来，下面是样例代码：

```python
import mxnet as mx
import mxnet.ndarray as nd
import gluoncv
from mxnet.gluon.model_zoo import vision

resnet18 = vision.resnet18_v1(pretrained=True)
print(resnet18)

# input image in x
filename = 'th.jpg'
img = mx.image.imread(filename)
x = gluoncv.data.transforms.presets.imagenet.transform_eval(img)

for i in range(len(resnet18.features)):
    layer = resnet18.features[i]
    x = layer(x)
    print(type(x),":",x.shape)
x = resnet18.output(x) #最后一层别忘记了
print(nd.topk(x, k=5))
```

可以看看  print(resnet18) 的输出：

```
ResNetV1(
  (output): Dense(512 -> 1000, linear)
  (features): HybridSequential(
    (0): Conv2D(3 -> 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm(momentum=0.9, use_global_stats=False, axis=1, eps=1e-05, fix_gamma=False, in_channels=64)
    (2): Activation(relu)
    (3): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False)
    (4): HybridSequential(
      (0): BasicBlockV1(
        (body): HybridSequential(
          (0): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm(momentum=0.9, use_global_stats=False, axis=1, eps=1e-05, fix_gamma=False, in_channels=64)
          (2): Activation(relu)
          (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm(momentum=0.9, use_global_stats=False, axis=1, eps=1e-05, fix_gamma=False, in_channels=64)
        )
      )
      (1): BasicBlockV1(
        (body): HybridSequential(
          (0): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm(momentum=0.9, use_global_stats=False, axis=1, eps=1e-05, fix_gamma=False, in_channels=64)
          (2): Activation(relu)
          (3): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm(momentum=0.9, use_global_stats=False, axis=1, eps=1e-05, fix_gamma=False, in_channels=64)
        )
      )
    )
    # ...这里省略几千字
    (8): GlobalAvgPool2D(size=(1, 1), stride=(1, 1), padding=(0, 0), ceil_mode=True)
  )
)
```

可以看到其架构不只是stack，还有层级关系，圆括号里带数字的基本上就是用下标访问，圆括号里带标识符（例如body），就通过属性名访问，例如：

```python
resnet18.features[4][0].body[2]
```

### 9、存储和恢复网络

相比caffe，mxnet的API比较乱，拼拼凑凑的。我了解到可以有好几种方式：

1. 使用python语言的pickle包dump和load。网络架构和参数一锅烩在输出文件里了。
2. 如果网络和网络中的层都是继承自gluon.HybridBlock，那么调用网络的成员函数export和load_checkpoint，输出json文件描述网络架构、param文件描述的是网络参数。这种方法的代码挺啰嗦的，不优雅，适用范围也比较局限，但输出的文件是比较标准的，例如mathematica可以加载。mathematica使用的底层就是mxnet
3. 使用网络的save_parameters()和load_parameters()成员函数保存和加载参数，但不能保存网络架构
4. 使用print(net, file)的方式打印出来的文件还挺可读的，看是不是自己写一个load反向生成网络架构

```python
symbol_file = os.path.join(ROOT_DIR, self.config.cp_dir, 'triplet-net')
self.model.export(path=symbol_file, epoch=epoch)  # gluon的export

prefix = os.path.join(ROOT_DIR, self.config.cp_dir, "triplet-net")  # export导出的前缀
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=prefix, epoch=5)
net = gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var('data'))  # 加载网络结构
# 设置网络参数
net_params = net.collect_params()
for param in arg_params:
    if param in net_params:
        net_params[param]._load_init(arg_params[param], ctx=ctx)
for param in aux_params:
    if param in net_params:
        net_params[param]._load_init(aux_params[param], ctx=ctx)
```

### 10、model zoo

gluon提供了很多预训练好的模型，包括alexnet、vgg、ssd等等。主要集中在cv和nlp方面：

```
https://gluon-cv.mxnet.io/model_zoo/index.html
https://gluon-nlp.mxnet.io/model_zoo/index.html
```

gluon会下载训练好的.param参数文件，直接使用训练好的模型，也可以基于已经训练好的模型和自己的数据做进一步的fine tuning。详细见上述文档中的HINT说明部分。



参数文件的默认保存的地址是~/.mxnet/models（windows下就是C:\Users\\\<username\>\\.mxnet\models）。

如果手动下载了该模型参数文件，可以放到这个目录下直接使用。

~/.mxnet目录是gluon的数据缓存目录，例如下载的mnist等知名数据集会保存在这个目录的datasets子目录下。

如果是使用mxnet的API，那么缓存目录可能是C:\\Users\\\<username\>\\AppData\\Roaming\\mxnet\\，有点混乱！



### 11、知名数据集的访问

对于一些知名数据集，都有对应的类可以直接使用，例如：

```
class gluoncv.data.ImageNet(root='~/.mxnet/datasets/imagenet', train=True, transform=None)

class mx.gluon.data.vision.MNIST()
class mx.gluon.data.vision.FashionMNIST()
class mx.gluon.data.vision.CIFAR10()
class mx.gluon.data.vision.CIFAR100()
```

更多知名数据集的访问类可以见：

```
https://gluon-cv.mxnet.io/api/data.datasets.html#id1
https://mxnet.incubator.apache.org/api/python/gluon/data.html
# mxnet 文档和接口挺乱的，上面就列了两个url，做同类事情
```

知名数据集下载下来默认保存在~/.mxnet/datasets目录下。

对于自定义的数据集，可以自己实现相关的类，也可以用mxnet提供的：

```
class mxnet.gluon.data.vision.datasets.ImageFolderDataset
```

### 12、pytorch

mxnet的gluon接口与pytorch的编程接口非常类似，我怀疑mxnet在设计gluon接口的时候，参考了pytorch的接口。差异稍微大一点的是autograd的设计。

几个对应的类/包 如下：

| mxnet                   | pytorch         |
| ----------------------- | --------------- |
| mxnet.gluon.NDArray     | torch.tensor    |
| mxnet.gluon.HybridBlock | torch.nn.Module |
| mxnet.gluon.nn          | torch.nn        |
| Dataset/DataLoader      | Dataset/DataLoader|
| gluon.Trainer | torch.optim.SGD |
| gluon.loss.SoftmaxCrossEntropyLoss | nn.CrossEntropyLoss |

下面是一个一段简单的softmax regression的pytorch代码：

```python
import torch
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt
import PIL.Image
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch.nn as nn

epoches = 10
batchsz = 10000
lr = 0.5
compute = False
file = './data/pytorch_softmax_reg.pk'

transform = transforms.Compose(
    [transforms.ToTensor()])
set1 = tv.datasets.MNIST("./data", True, download=True, transform=transform)
train_data = dataloader.DataLoader(set1, batchsz, True)# type:dataloader.DataLoader
set1 = tv.datasets.MNIST("./data", False, download=True, transform=transform)
test_data = dataloader.DataLoader(set1, batchsz, True)# type:dataloader.DataLoader

def test(model, test_data):
    s = nn.Softmax(dim=1)
    for images, labels in test_data:
        images = images.reshape(batchsz, -1).to(device="cuda:0")
        labels = labels.to(device="cuda:0")
        y = model(images)
        y = s(y)
        y = torch.max(y, 1) # type:torch.return_types.max
        same = (y[1] == labels).sum()
        return same.to(device='cpu').numpy() / batchsz

model = nn.Linear(28*28, 10).to(device="cuda:0")
trainer = torch.optim.SGD(model.parameters(), lr)
lossfun = nn.CrossEntropyLoss()

if compute:
    for e in range(epoches):
        for images, labels in train_data:
            images = images.reshape(batchsz, -1).to(device="cuda:0")
            labels = labels.to(device="cuda:0")
            y = model(images)
            trainer.zero_grad()
            L = lossfun(y, labels)
            L.backward()
            trainer.step()
        print(L)
        print(test(model, test_data))
    torch.save(model.state_dict(), file)
else:
    model = nn.Linear(28*28, 10).to(device="cuda:0")
    model.load_state_dict(torch.load(file))
    print(test(model, test_data))
```



### 坑

1、ndarray之间的相互运算，例如两个nd相加，他们的dtype要求一致，否则报错，类似这样的：

```
Incompatible attr in node  at 1-th input: expected int32, got float32
```

### 不错的文档

```
http://zh.gluon.ai/index.html
https://gluon.mxnet.io/
```