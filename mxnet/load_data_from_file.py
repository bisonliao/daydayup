"""
自定义读取训练数据的类，主要是两个函数getitem和len
"""
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



