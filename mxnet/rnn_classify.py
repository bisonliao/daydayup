'''
让rnn学会对一个数学表达式的结果正负做判断
例如 "1+2"分类为正  "4-9"分类为负
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

#####  grp #1 acc = 100% ##################################
'''batch_size = 64
num_batches = 100
args_lr = 0.001
args_epochs = 50
max_added = 100  # 加数的最大值，很关键

# observe the loss value ,then modify the get_lr to work properly
def get_lr(ep):
    if ep < 10:
        return args_lr
    elif ep >= 10 and ep < 40:
        return args_lr / 10
    elif ep >=40:
        return args_lr / 100'''


###### grp #2 acc = 99% ####################################
batch_size = 64
num_batches = 100
args_lr = 0.001
args_epochs = 50
max_added = 255  # 加数的最大值，很关键

# observe the loss value ,then modify the get_lr to work properly
def get_lr(ep):
    if ep < 10:
        return args_lr
    elif ep >= 10 and ep < 40:
        return args_lr / 10
    elif ep >=40:
        return args_lr / 40

###########################################################
context = mx.gpu(0)
vocab_size = 12  # 0-9 and + -
seq_length = 16
args_model = 'lstm'
args_nhid = 512
args_nlayers = 5
args_clip = 0.2
args_dropout = 0.0001
args_save = './data/rnn_classify.param'





###############################################
def char_to_vect():
    vect = nd.zeros((vocab_size,vocab_size))
    for i in range(vect.shape[0]):
        vect[i][i] = 1
    result = {}
    i = 0
    for c in b"0123456789+-":
        result[chr(c)] = vect[i]
        i +=1
    return result
def vect_to_char(v):
    index = nd.argmax(v,axis=0).asscalar()
    chars = b"0123456789+-"

    return chr(chars[int(index)])

c2v = char_to_vect()



# genenerate train dataset ,shape is  [batch_number, seq_len, batch_sz, input_size]
def gene_data():


    num_samples = num_batches *  batch_size
    labels = []
    data=nd.zeros((num_samples, seq_length, vocab_size))
    neg_cnt = 0
    pos_cnt = 0
    for i in range(num_samples):
        a = np.random.randint(0, max_added)
        b = np.random.randint(0, max_added)
        op = np.random.randint(0,3) # 0,1,2
        if op == 2:
            op = '+'
            c = a+b
        else :
            op = '-'
            c = a - b


        inputstr = "%d%s%d"%(a,op,b)
        inputstrlen = len(inputstr)
        for j in range(seq_length - inputstrlen):
            inputstr = "0" + inputstr
        if c >= 0:
            labels.append(0)
            pos_cnt+=1
        else:
            labels.append(1)
            neg_cnt +=1
        for j in range(len(inputstr)):
            data[i,j] = c2v[ inputstr[j]]

    print("pos/neg count:", pos_cnt, ", ", neg_cnt)


    data = data.reshape((num_samples, seq_length, vocab_size))
    train_data = data.reshape((num_batches, batch_size, seq_length, vocab_size))
    # swap batch_size and seq_length axis to make later access easier
    train_data = nd.swapaxes(train_data, 1, 2)

    labels = nd.array(labels)
    train_label = labels.reshape((num_batches,1, batch_size))
    return (train_data[0:-10].as_in_context(context),
            train_label[0:-10].as_in_context(context),
            train_data[-10:].as_in_context(context),
            train_label[-10:].as_in_context(context))



train_data, train_label,test_data, test_label = gene_data()

# check the  relation between data and label
def check_data(data, label):
    leftstr = ""
    for i in range(seq_length):
        leftstr += vect_to_char(data[3, i, 7])
    print(leftstr, ":", (label[3,0,7]).asscalar())


check_data(train_data, train_label)
check_data(test_data, test_label)




###############################################
# define the rnn model
class RNNModel(gluon.Block):
    def __init__(self, mode, input_sz, num_hidden,
                 num_layers, dropout=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():

            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=input_sz)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=input_sz)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=input_sz)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=input_sz)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            # 加不加这一层效果差不太多，加了这层会好几个百分点
            self.fc = nn.Dense(512, in_units=num_hidden)
            self.output = nn.Dense(2, in_units=512)
            #self.output = nn.Dense(1, in_units=num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        output = output[-1]
        output = self.fc(output.reshape((-1, self.num_hidden)))
        output = self.output(output)
        return output, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

model = RNNModel(args_model, vocab_size, args_nhid,
                       args_nlayers, args_dropout)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'adam',
                        {'learning_rate': args_lr, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# detach SDG graph，avoid backward
def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden



# get the test accuracy

def evaluate_accuracy(test_data,test_label, net):
    acc = mx.metric.Accuracy()
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    for  i in range(test_data.shape[0]):
        data = test_data[i]
        label = test_label[i]
        label = label.reshape((batch_size,))
        output, hidden = net(data, hidden)
        output = nd.softmax(output)
        predictions = output.argmax(axis=1)
        acc.update(preds=predictions, labels=label)
        if (i == 0) :
            print(predictions, "\n", label)
    em = acc.get()
    return em[1]

########################################################
# train the model
def train():
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()

        trainer.set_learning_rate(get_lr(epoch))

        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
        for  i in range(train_data.shape[0]):
            data = train_data[i]
            target = train_label[i]
            target = target.reshape((batch_size,))


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



        cur_L = total_L / seq_length / batch_size / num_batches
        print('loss %f, %f [Epoch %d ]' % ( cur_L,cur_L * 1000000,epoch + 1))


    val_acc = evaluate_accuracy(test_data, test_label,model)

    print(' validation accuracy %.2f, validation perplexity %.2f' % (val_acc, math.exp(val_acc)))

train()

