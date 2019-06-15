### 多批次的梯度累加

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

### 学习率、学习率倍数等超参数

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

### 自定义的数据读取方式

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



### 自定义layer

gluon.Block是mxnet里非常重要的类，重要性与NDArray不相上下。

自定义layer就是写一个继承Block的子类，具体做法：

```
https://gluon.mxnet.io/chapter03_deep-neural-networks/custom-layer.html#Defining-a-(toy)-custom-layer
```

### Train mode and predict mode

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

### 获取中间层的输出

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