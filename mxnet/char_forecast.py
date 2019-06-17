'''
使用rnn根据一篇英文文章，训练出一个模型，根据前面的字符输入，预测下一个字符
共54个字符，准确率59%
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
args_bptt = 5
args_dropout = 0.2
args_save = 'char_forecast.param'

###############################################
#generate a one-hot matrix
def one_hots(numerical_list, vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=context)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

# convert one-hot embeding code to readable text
def textify(embedding, character_list):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
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
    time_numerical = [character_dict[char] for char in nietzsche]

    '''
    这样写是有问题的
    num_samples = (len(time_numerical) - 1) // seq_length
    num_batches = num_samples // batch_size

    train_data = one_hots(time_numerical[:seq_length * batch_size * num_batches],vocab_size) #.reshape((num_batches, seq_length, batch_size, vocab_size))
    train_label = one_hots(time_numerical[1:seq_length * batch_size * num_batches + 1],vocab_size)#.reshape((num_batches, seq_length, batch_size, vocab_size))
    '''
    # -1 here so we have enough characters for labels later
    num_samples = (len(time_numerical) - 1) // seq_length
    dataset = one_hots(time_numerical[:seq_length * num_samples],vocab_size).reshape((num_samples, seq_length, vocab_size))
    num_batches = len(dataset) // batch_size
    train_data = dataset[:num_batches * batch_size].reshape((num_batches, batch_size, seq_length, vocab_size))
    # swap batch_size and seq_length axis to make later access easier
    train_data = nd.swapaxes(train_data, 1, 2)

    labels = one_hots(time_numerical[1:seq_length * num_samples + 1],vocab_size)
    train_label = labels.reshape((num_batches, batch_size, seq_length, vocab_size))
    train_label = nd.swapaxes(train_label, 1, 2)
    return (train_data[0:-10],train_label[0:-10],train_data[-10:],train_label[-10:])


if False :
    dataset = gene_data()
    with open("./data/char_forecast.data", "wb") as f:
        pickle.dump(dataset, f)
else:
    with open("./data/char_forecast.data", "rb") as f:
        dataset = pickle.load(f)

train_data, train_label,test_data, test_label = dataset

# check the  relation between data and label
def check_data(data, label):
    tmplist = nd.zeros((seq_length,vocab_size))
    for i in range(seq_length):
        tmplist[i] = data[3, i, 7]
    print(textify(tmplist, character_list))
    print("-----------")
    tmplist = nd.zeros((seq_length,vocab_size))
    for i in range(seq_length):
        tmplist[i] = label[3, i, 7]
    print(textify(tmplist, character_list))
    print("-----------")

check_data(train_data, train_label)
check_data(test_data, test_label)


###############################################
# define the rnn model
class RNNModel(gluon.Block):
    def __init__(self, mode, vocab_size, num_embed, num_hidden,
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
            self.output = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        output = self.output(output.reshape((-1, self.num_hidden)))
        return output, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

model = RNNModel(args_model, vocab_size, vocab_size, args_nhid,
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
        label = nd.argmax(label, axis=2)
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
        label = nd.argmax(label, axis=2)
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
            target = nd.argmax(target, axis=2)
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

            show_interval = 1000
            if i % show_interval == 0 and i > 0:
                cur_L = total_L / seq_length / batch_size / show_interval
                print('[Epoch %d Batch %d] loss %.2f' % ( epoch + 1, i, cur_L))
                total_L = 0.0

        val_acc = evaluate_accuracy(test_data, test_label,model)

        print('[Epoch %d] time cost %.2fs, validation accuracy %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_acc, math.exp(val_acc)))

train()