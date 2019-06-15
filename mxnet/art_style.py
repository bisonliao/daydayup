'''
用mxnet实现对一张照片做艺术风格化处理
先用训练好的alexnet对梵高的星空做前向传播，提取其中的feature map作为ground truth
然后输入待处理的照片，迭代修改照片的像素，使得其feature map接近groud truth
'''
import mxnet as mx
import mxnet.ndarray as nd
import gluoncv
import mxnet.gluon as gluon
from mxnet.gluon.model_zoo import vision
import mxnet.autograd as autograd
import numpy as np
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt

context = mx.gpu(0)
epochs = 500
lr = 50

#####################################################
# get pretrained alxenet, params saved at dir model
net = vision.alexnet(pretrained = True,ctx=context,root="./model")
#print(net)
art = "./art.jpg"
img = mx.image.imread(art)
# apply default data preprocessing
transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)

def get_feature(net, x):
    for i in range(7):  # get the output of layer6 as groundtruth
        layer = net.features[i]
        x = layer(x)
    return x

#####################################################
# get ground truth
x = transformed_img.as_in_context(context)
groundtruth = get_feature(net, x)
print("ground truth shape:", groundtruth.shape)
#nd.save("./art_gt.ndarray", groundtruth)

#####################################################
# gradient desending, modify the input
photo = "./cat.jpg"
img = mx.image.imread(photo)
# apply default data preprocessing
transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
x = transformed_img.as_in_context(context)
loss_fn = gluon.loss.L2Loss()

for e in range(epochs):
    x.attach_grad()
    with autograd.record(True):
        output = get_feature(net, x)
        loss = loss_fn(output,groundtruth)
    loss.backward()
    #print(x.grad)
    print("epoch:", e, " loss:", loss.asscalar())
    x = x - lr * x.grad

#####################################################
#show modified picture

#min-max归一化
def scale(data):
    maxvalue = nd.max(data).asscalar()
    minvalue = nd.min(data).asscalar()
    length = maxvalue - minvalue
    return (data - minvalue) / length


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
    return data

pic = scale(x[0])
pic = chw2hwc(pic)
plt.imshow(pic.asnumpy())
plt.show()

