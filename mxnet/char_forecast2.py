'''
使用rnn根据一篇英文文章，训练出一个模型，根据前面的字符输入，预测下一个字符
与char_forecast.py不一样的是：没有将字符编码为one-hot，而是直接使用其在集合中的顺序号（输入data做了归一化，否则准确率更低）
54个字符（分类数），最后准确率51%
'''
import mxnet as mx
import mxnet.ndarray as nd
import gluoncv
import mxnet.gluon as gluon
from mxnet.gluon.model_zoo import vision
import mxnet.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import mxnet.gluon.nn as nn
import mxnet.gluon.rnn as rnn
import time
import math
import pickle

context = mx.gpu(0)
batch_size = 32
seq_length = 64
args_model = 'lstm'
args_nhid = 100
args_nlayers = 2
args_lr = 0.1
args_clip = 0.2
args_epochs = 100
args_dropout = 0.2
args_save = 'char_forecast.param'

###############################################

# convert one-hot embeding code to readable text
def textify(embedding, character_list, isLabel):
    result = ""
    if isLabel:
        indices = embedding.asnumpy()
    else:
        indices = embedding.asnumpy() *vocab_size
    for idx in indices:
        result += chr(character_list[int(idx)])
    return result

# clean the text, only alpha and space is left
def clean(text):
    result=bytes()
    for i in range(len(text)):
        c = chr(text[i])
        if c.isalpha() or c == ' ':
            result = result + text[i:i+1]
        else:
            result = result + b' '
    return result

with open("./data/nietzsche.txt", "rb") as f:
    nietzsche = f.read()
nietzsche = clean(nietzsche)
print(nietzsche[:50])
character_list = list(set(nietzsche)) #distinct char
vocab_size = len(character_list)
print("voc_size:", vocab_size)

# genenerate train dataset ,shape is  [batch_number, seq_len, batch_sz, input_size]
def gene_data():

    character_dict = {}
    for e, char in enumerate(character_list):
        character_dict[char] = e
    time_numerical = [character_dict[char] / vocab_size   for char in nietzsche]
    time_numerical2 = [character_dict[char]  for char in nietzsche]


    # -1 here so we have enough characters for labels later
    num_samples = (len(time_numerical) - 1) // seq_length
    dataset = nd.array(time_numerical[:seq_length * num_samples], ctx=context).reshape((num_samples, seq_length, 1))
    num_batches = len(dataset) // batch_size
    train_data = dataset[:num_batches * batch_size].reshape((num_batches, batch_size, seq_length,1))
    # swap batch_size and seq_length axis to make later access easier
    train_data = nd.swapaxes(train_data, 1, 2)

    labels = nd.array(time_numerical2[1:seq_length * num_samples + 1],ctx=context)
    train_label = labels.reshape((num_batches, batch_size, seq_length,1))
    train_label = nd.swapaxes(train_label, 1, 2)
    return (train_data[0:-10],train_label[0:-10],train_data[-10:],train_label[-10:])



train_data, train_label,test_data, test_label = gene_data()



# check the  relation between data and label
def check_data(data, label):
    tmplist = nd.zeros((seq_length,))
    for i in range(seq_length):
        tmplist[i] = data[3, i, 7]
    print(textify(tmplist, character_list, False))
    print("-----------")
    tmplist = nd.zeros((seq_length,))
    for i in range(seq_length):
        tmplist[i] = label[3, i, 7]
    print(textify(tmplist, character_list, True))
    print("-----------")

check_data(train_data, train_label)
check_data(test_data, test_label)


###############################################
# define the rnn model
class RNNModel(gluon.Block):
    def __init__(self, mode, class_num, num_embed, num_hidden,
                 num_layers, dropout=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():

            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            self.output = nn.Dense(class_num, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        output = self.output(output.reshape((-1, self.num_hidden)))
        return output, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

model = RNNModel(args_model, vocab_size, 1, args_nhid,
                       args_nlayers, args_dropout)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# detach SDG graph，avoid backward
def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

# get the test loss
def eval(test_data,test_label):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx=context)
    for i in range(test_data.shape[0]):
        data = test_data[i]
        label = test_label[i]
        label = label.reshape((seq_length * batch_size,))
        output, hidden = model(data, hidden)
        L = loss(output, label)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

# get the test accuracy
def evaluate_accuracy(test_data,test_label, net):
    acc = mx.metric.Accuracy()
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    for  i in range(test_data.shape[0]):
        data = test_data[i]
        label = test_label[i]
        label = label.reshape((seq_length * batch_size,))
        output, hidden = net(data, hidden)
        output = nd.softmax(output)
        predictions = output.argmax(axis=1)
        acc.update(preds=predictions, labels=label)
    em = acc.get()
    return em[1]

########################################################
# train the model
def train():
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)

        for  i in range(train_data.shape[0]):
            data = train_data[i]
            target = train_label[i]
            target = target.reshape((seq_length*batch_size,))

            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args_clip * seq_length * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            show_interval = 100
            if i % show_interval == 0 and i > 0:
                cur_L = total_L / seq_length / batch_size / show_interval
                print('[Epoch %d Batch %d] loss %.2f' % ( epoch + 1, i, cur_L))
                total_L = 0.0


        val_acc = evaluate_accuracy(test_data, test_label,model)
        print('[Epoch %d] time cost %.2fs, validation accuracy %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_acc, math.exp(val_acc)))

train()