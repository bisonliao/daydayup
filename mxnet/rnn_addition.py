'''
让rnn学会做整数加法
最后只能学习一个模型，计算出比较接近的两个数的和
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
batch_size = 64
num_batches = 100
vocab_size = 11  # 0-9 and +
seq_length = 16
args_model = 'lstm'
args_nhid = 512
args_nlayers = 5
args_lr = 0.001
args_clip = 0.2
args_epochs = 100
args_dropout = 0.0001
args_save = 'rnn_addition.param'

###############################################
def char_to_vect():
    vect = nd.zeros((vocab_size,vocab_size))
    for i in range(vect.shape[0]):
        vect[i][i] = 1
    result = {}
    i = 0
    for c in b"0123456789+":
        result[chr(c)] = vect[i]
        i +=1
    return result
def vect_to_char(v):
    index = nd.argmax(v,axis=0).asscalar()
    chars = b"0123456789+"

    return chr(chars[int(index)])

c2v = char_to_vect()



# genenerate train dataset ,shape is  [batch_number, seq_len, batch_sz, input_size]
def gene_data():


    num_samples = num_batches *  batch_size
    labels = []
    data=nd.zeros((num_samples, seq_length, vocab_size))
    for i in range(num_samples):
        a = np.random.randint(0, 65535)
        b = np.random.randint(0, 65535)
        c = a+b
        inputstr = "%08d+%07d"%(a,b)
        inputstrlen = len(inputstr)
        for j in range(seq_length - inputstrlen):
            inputstr = "0" + inputstr

        labels.append(c/65535)
        for j in range(len(inputstr)):
            data[i,j] = c2v[ inputstr[j]]


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
    print(leftstr, "=", (label[3,0,7]*65535).asscalar())


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
            self.fc = nn.Dense(512, in_units=num_hidden)
            self.output = nn.Dense(1, in_units=512)
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
loss = gluon.loss.L2Loss()

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
    r = 0
    c = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    for  i in range(test_data.shape[0]):
        data = test_data[i]
        label = test_label[i]
        label = label.reshape((batch_size,))
        output, hidden = net(data, hidden)
        output = output.reshape((batch_size,))
        output = nd.round(output*65535)
        label = nd.round(label*65535)
        for j in range(batch_size):
            print(output[j].asscalar(), " vs ", label[j].asscalar())
            c += 1
            if (output[j].asscalar() == label[j].asscalar()):
                r += 1
    return r / c

########################################################
# train the model
def train():
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()

        if epoch == 40:
            trainer.set_learning_rate(args_lr / 40)

        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
        for  i in range(train_data.shape[0]):
            data = train_data[i]
            target = train_label[i]
            target = target.reshape((batch_size,))


            hidden = detach(hidden)

            with autograd.record():
                output, hidden = model(data, hidden)
                output = output.reshape((batch_size,))
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args_clip * seq_length * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            show_interval = 50
            if i % show_interval == 0 and i > 0:
                cur_L = total_L / seq_length / batch_size / show_interval
                print('loss %f [Epoch %d Batch %d]' % ( cur_L,epoch + 1, i))
                total_L = 0.0

    val_acc = evaluate_accuracy(test_data, test_label,model)

    print(' validation accuracy %.2f, validation perplexity %.2f' % (val_acc, math.exp(val_acc)))

train()

'''
这里保存最好的一次结果 ：）
E:\install\Python\Python35\python.exe E:/DeepLearning/mxnet/pyproject/test1/rnn_addition.py
0000054343+22261 = 
[76604.]
<NDArray 1 @gpu(0)>
00000000620+3538 = 
[4158.]
<NDArray 1 @gpu(0)>
loss 0.008880 [Epoch 1 Batch 50]
loss 0.003336 [Epoch 2 Batch 50]
loss 0.001585 [Epoch 3 Batch 50]
loss 0.000776 [Epoch 4 Batch 50]
loss 0.000466 [Epoch 5 Batch 50]
loss 0.000421 [Epoch 6 Batch 50]
loss 0.000242 [Epoch 7 Batch 50]
loss 0.000235 [Epoch 8 Batch 50]
loss 0.000143 [Epoch 9 Batch 50]
loss 0.000131 [Epoch 10 Batch 50]
loss 0.000132 [Epoch 11 Batch 50]
loss 0.000081 [Epoch 12 Batch 50]
loss 0.000079 [Epoch 13 Batch 50]
loss 0.000070 [Epoch 14 Batch 50]
loss 0.000056 [Epoch 15 Batch 50]
loss 0.000052 [Epoch 16 Batch 50]
loss 0.000037 [Epoch 17 Batch 50]
loss 0.000047 [Epoch 18 Batch 50]
loss 0.000035 [Epoch 19 Batch 50]
loss 0.000033 [Epoch 20 Batch 50]
loss 0.000025 [Epoch 21 Batch 50]
loss 0.000019 [Epoch 22 Batch 50]
loss 0.000018 [Epoch 23 Batch 50]
loss 0.000021 [Epoch 24 Batch 50]
loss 0.000024 [Epoch 25 Batch 50]
loss 0.000025 [Epoch 26 Batch 50]
loss 0.000022 [Epoch 27 Batch 50]
loss 0.000019 [Epoch 28 Batch 50]
loss 0.000018 [Epoch 29 Batch 50]
loss 0.000019 [Epoch 30 Batch 50]
loss 0.000020 [Epoch 31 Batch 50]
loss 0.000024 [Epoch 32 Batch 50]
loss 0.000021 [Epoch 33 Batch 50]
loss 0.000019 [Epoch 34 Batch 50]
loss 0.000016 [Epoch 35 Batch 50]
loss 0.000013 [Epoch 36 Batch 50]
loss 0.000011 [Epoch 37 Batch 50]
loss 0.000007 [Epoch 38 Batch 50]
loss 0.000005 [Epoch 39 Batch 50]
loss 0.000007 [Epoch 40 Batch 50]
loss 0.000002 [Epoch 41 Batch 50]
loss 0.000002 [Epoch 42 Batch 50]
loss 0.000002 [Epoch 43 Batch 50]
loss 0.000002 [Epoch 44 Batch 50]
loss 0.000002 [Epoch 45 Batch 50]
loss 0.000002 [Epoch 46 Batch 50]
loss 0.000002 [Epoch 47 Batch 50]
loss 0.000002 [Epoch 48 Batch 50]
loss 0.000002 [Epoch 49 Batch 50]
loss 0.000002 [Epoch 50 Batch 50]
loss 0.000001 [Epoch 51 Batch 50]
loss 0.000002 [Epoch 52 Batch 50]
loss 0.000001 [Epoch 53 Batch 50]
loss 0.000001 [Epoch 54 Batch 50]
loss 0.000001 [Epoch 55 Batch 50]
loss 0.000001 [Epoch 56 Batch 50]
loss 0.000001 [Epoch 57 Batch 50]
loss 0.000001 [Epoch 58 Batch 50]
loss 0.000001 [Epoch 59 Batch 50]
loss 0.000001 [Epoch 60 Batch 50]
loss 0.000001 [Epoch 61 Batch 50]
loss 0.000001 [Epoch 62 Batch 50]
loss 0.000001 [Epoch 63 Batch 50]
loss 0.000001 [Epoch 64 Batch 50]
loss 0.000001 [Epoch 65 Batch 50]
loss 0.000001 [Epoch 66 Batch 50]
loss 0.000001 [Epoch 67 Batch 50]
loss 0.000001 [Epoch 68 Batch 50]
loss 0.000001 [Epoch 69 Batch 50]
loss 0.000001 [Epoch 70 Batch 50]
loss 0.000001 [Epoch 71 Batch 50]
loss 0.000001 [Epoch 72 Batch 50]
loss 0.000001 [Epoch 73 Batch 50]
loss 0.000001 [Epoch 74 Batch 50]
loss 0.000001 [Epoch 75 Batch 50]
loss 0.000001 [Epoch 76 Batch 50]
loss 0.000001 [Epoch 77 Batch 50]
loss 0.000001 [Epoch 78 Batch 50]
loss 0.000001 [Epoch 79 Batch 50]
loss 0.000001 [Epoch 80 Batch 50]
loss 0.000001 [Epoch 81 Batch 50]
loss 0.000001 [Epoch 82 Batch 50]
loss 0.000001 [Epoch 83 Batch 50]
loss 0.000001 [Epoch 84 Batch 50]
loss 0.000001 [Epoch 85 Batch 50]
loss 0.000001 [Epoch 86 Batch 50]
loss 0.000001 [Epoch 87 Batch 50]
loss 0.000001 [Epoch 88 Batch 50]
loss 0.000001 [Epoch 89 Batch 50]
loss 0.000001 [Epoch 90 Batch 50]
loss 0.000001 [Epoch 91 Batch 50]
loss 0.000001 [Epoch 92 Batch 50]
loss 0.000001 [Epoch 93 Batch 50]
loss 0.000001 [Epoch 94 Batch 50]
loss 0.000001 [Epoch 95 Batch 50]
loss 0.000001 [Epoch 96 Batch 50]
loss 0.000001 [Epoch 97 Batch 50]
loss 0.000001 [Epoch 98 Batch 50]
loss 0.000001 [Epoch 99 Batch 50]
loss 0.000001 [Epoch 100 Batch 50]
76539.0  vs  76963.0
89203.0  vs  88912.0
44952.0  vs  44760.0
86481.0  vs  86256.0
102598.0  vs  102660.0
45187.0  vs  44935.0
83453.0  vs  83704.0
38319.0  vs  37866.0
100061.0  vs  99714.0
96543.0  vs  96197.0
68163.0  vs  68329.0
112234.0  vs  111856.0
53265.0  vs  54438.0
123587.0  vs  124298.0
72759.0  vs  73245.0
87342.0  vs  87565.0
28104.0  vs  28034.0
92242.0  vs  92159.0
64798.0  vs  64556.0
61765.0  vs  62193.0
99905.0  vs  100455.0
70088.0  vs  69384.0
51188.0  vs  53540.0
27988.0  vs  27979.0
60451.0  vs  60198.0
65128.0  vs  65352.0
37763.0  vs  38062.0
21992.0  vs  22639.0
60733.0  vs  60397.0
104740.0  vs  103805.0
18961.0  vs  19297.0
88096.0  vs  88188.0
27824.0  vs  28325.0
26597.0  vs  26342.0
52138.0  vs  52264.0
41266.0  vs  41126.0
83461.0  vs  83370.0
88375.0  vs  88220.0
66616.0  vs  66557.0
93144.0  vs  93017.0
42501.0  vs  42641.0
90083.0  vs  89992.0
21121.0  vs  21101.0
61358.0  vs  60448.0
51641.0  vs  51298.0
82685.0  vs  82807.0
16003.0  vs  15557.0
98278.0  vs  98016.0
93740.0  vs  93529.0
69189.0  vs  68395.0
70244.0  vs  69888.0
25040.0  vs  24629.0
54117.0  vs  54792.0
81635.0  vs  81406.0
106808.0  vs  106402.0
87928.0  vs  87521.0
84365.0  vs  84377.0
56387.0  vs  56381.0
126521.0  vs  126457.0
114697.0  vs  114284.0
47636.0  vs  47210.0
115075.0  vs  114491.0
77007.0  vs  76917.0
82384.0  vs  82176.0
46116.0  vs  45887.0
118973.0  vs  119249.0
82561.0  vs  82315.0
94219.0  vs  94005.0
61240.0  vs  60755.0
67477.0  vs  67289.0
91365.0  vs  91178.0
53622.0  vs  53661.0
94461.0  vs  94264.0
48776.0  vs  49804.0
80869.0  vs  80650.0
95199.0  vs  95269.0
38079.0  vs  38403.0
45902.0  vs  46182.0
52684.0  vs  52771.0
39585.0  vs  39307.0
47299.0  vs  47650.0
51062.0  vs  51236.0
79877.0  vs  79870.0
101431.0  vs  101270.0
38751.0  vs  38635.0
71593.0  vs  71861.0
38891.0  vs  39246.0
75425.0  vs  75171.0
55995.0  vs  56310.0
94589.0  vs  94403.0
56709.0  vs  56677.0
63196.0  vs  62646.0
53950.0  vs  53793.0
30699.0  vs  31140.0
58372.0  vs  58611.0
51196.0  vs  52586.0
57629.0  vs  57840.0
4618.0  vs  4663.0
71748.0  vs  71488.0
53687.0  vs  53189.0
95381.0  vs  95338.0
76929.0  vs  77043.0
82607.0  vs  82457.0
29890.0  vs  30072.0
105687.0  vs  105474.0
75263.0  vs  75261.0
51386.0  vs  51547.0
42831.0  vs  42329.0
51031.0  vs  51195.0
75605.0  vs  75687.0
42927.0  vs  43293.0
80834.0  vs  81133.0
90395.0  vs  90330.0
82435.0  vs  82326.0
44871.0  vs  44901.0
62901.0  vs  62912.0
22732.0  vs  23029.0
100707.0  vs  100492.0
74728.0  vs  74727.0
91297.0  vs  90992.0
38836.0  vs  38349.0
113955.0  vs  113959.0
59490.0  vs  59706.0
76945.0  vs  76944.0
99373.0  vs  99056.0
66151.0  vs  66204.0
66446.0  vs  66146.0
65108.0  vs  65034.0
95438.0  vs  95152.0
20139.0  vs  19185.0
97266.0  vs  97304.0
65168.0  vs  64977.0
99572.0  vs  99354.0
45490.0  vs  45653.0
39221.0  vs  39113.0
102466.0  vs  102650.0
82764.0  vs  82743.0
60886.0  vs  60639.0
65641.0  vs  65895.0
125273.0  vs  125726.0
6403.0  vs  5937.0
61109.0  vs  60310.0
70752.0  vs  70800.0
90699.0  vs  91127.0
73475.0  vs  73184.0
32886.0  vs  33334.0
71097.0  vs  70967.0
54127.0  vs  54478.0
91045.0  vs  90809.0
43761.0  vs  44280.0
117855.0  vs  118564.0
14048.0  vs  12825.0
98267.0  vs  97957.0
93486.0  vs  92820.0
22445.0  vs  21430.0
21291.0  vs  20754.0
48990.0  vs  48832.0
22768.0  vs  23210.0
85044.0  vs  84502.0
111973.0  vs  112170.0
97758.0  vs  97419.0
51468.0  vs  51463.0
39532.0  vs  37404.0
94699.0  vs  94375.0
80708.0  vs  80997.0
56738.0  vs  57004.0
55536.0  vs  55409.0
9698.0  vs  10435.0
46793.0  vs  46522.0
116954.0  vs  116384.0
105970.0  vs  105495.0
67687.0  vs  67476.0
62306.0  vs  62172.0
56261.0  vs  56477.0
97737.0  vs  97857.0
35442.0  vs  35972.0
10440.0  vs  11311.0
47945.0  vs  48360.0
52774.0  vs  52623.0
117075.0  vs  116692.0
48805.0  vs  47967.0
115699.0  vs  115677.0
60008.0  vs  59537.0
121765.0  vs  121491.0
16328.0  vs  16029.0
49828.0  vs  49469.0
23892.0  vs  23212.0
37835.0  vs  37427.0
54601.0  vs  54240.0
29429.0  vs  29467.0
63535.0  vs  63113.0
48001.0  vs  48137.0
47158.0  vs  47291.0
38404.0  vs  38486.0
69958.0  vs  69421.0
60290.0  vs  60482.0
109067.0  vs  108633.0
108911.0  vs  108640.0
71913.0  vs  71652.0
3115.0  vs  4158.0
80696.0  vs  80618.0
9031.0  vs  8329.0
61187.0  vs  61219.0
52173.0  vs  52105.0
118192.0  vs  117647.0
56466.0  vs  55964.0
50535.0  vs  50678.0
78826.0  vs  78781.0
61151.0  vs  60825.0
111328.0  vs  111184.0
9075.0  vs  9664.0
71375.0  vs  71267.0
46960.0  vs  46891.0
82163.0  vs  82346.0
88756.0  vs  88797.0
122276.0  vs  121945.0
82469.0  vs  82497.0
52102.0  vs  50424.0
57060.0  vs  57138.0
102335.0  vs  102540.0
90408.0  vs  90649.0
24305.0  vs  23989.0
4810.0  vs  5097.0
39154.0  vs  39185.0
65646.0  vs  65728.0
99010.0  vs  99099.0
13106.0  vs  11016.0
77928.0  vs  77802.0
64420.0  vs  64280.0
58214.0  vs  57725.0
22353.0  vs  21602.0
49918.0  vs  49490.0
20260.0  vs  20549.0
97636.0  vs  97610.0
17516.0  vs  16895.0
80134.0  vs  80308.0
97678.0  vs  97370.0
22085.0  vs  22258.0
99748.0  vs  99161.0
96903.0  vs  96794.0
7860.0  vs  8186.0
41288.0  vs  41284.0
75695.0  vs  76097.0
103980.0  vs  103246.0
105274.0  vs  105542.0
125346.0  vs  125431.0
52700.0  vs  52590.0
61489.0  vs  61929.0
106104.0  vs  105996.0
40343.0  vs  40104.0
30222.0  vs  29739.0
72371.0  vs  72275.0
85615.0  vs  85781.0
64726.0  vs  64835.0
39632.0  vs  39150.0
76362.0  vs  76113.0
41027.0  vs  40927.0
51452.0  vs  51876.0
43072.0  vs  43316.0
94241.0  vs  94581.0
26078.0  vs  25757.0
82539.0  vs  82737.0
48582.0  vs  47716.0
70844.0  vs  70761.0
70198.0  vs  70329.0
79886.0  vs  79871.0
38013.0  vs  38023.0
68848.0  vs  68921.0
33974.0  vs  32985.0
48035.0  vs  48217.0
110077.0  vs  109858.0
68443.0  vs  68655.0
17005.0  vs  17852.0
111283.0  vs  111198.0
106151.0  vs  105556.0
76126.0  vs  75614.0
78006.0  vs  78247.0
69517.0  vs  69630.0
42098.0  vs  43094.0
125229.0  vs  125868.0
29274.0  vs  29132.0
22260.0  vs  22142.0
18287.0  vs  17573.0
68795.0  vs  68704.0
95806.0  vs  95697.0
36939.0  vs  36865.0
115332.0  vs  114882.0
53209.0  vs  52823.0
87089.0  vs  86942.0
68966.0  vs  69177.0
72474.0  vs  72160.0
54967.0  vs  54699.0
74495.0  vs  74060.0
105216.0  vs  104941.0
45500.0  vs  45676.0
71818.0  vs  71736.0
56301.0  vs  55945.0
82530.0  vs  82672.0
61598.0  vs  61515.0
74418.0  vs  74857.0
86841.0  vs  86479.0
27405.0  vs  26934.0
68677.0  vs  68670.0
64200.0  vs  63414.0
54533.0  vs  54569.0
67870.0  vs  68008.0
61419.0  vs  60753.0
59579.0  vs  59271.0
55812.0  vs  55437.0
69618.0  vs  69502.0
30062.0  vs  30022.0
27786.0  vs  27888.0
70449.0  vs  70093.0
77941.0  vs  78301.0
63698.0  vs  63762.0
122133.0  vs  121765.0
68026.0  vs  67959.0
85361.0  vs  85517.0
35895.0  vs  36521.0
83650.0  vs  83157.0
82644.0  vs  82805.0
7282.0  vs  7147.0
40494.0  vs  40174.0
42773.0  vs  43279.0
63392.0  vs  63057.0
53499.0  vs  53762.0
43150.0  vs  43042.0
113445.0  vs  113035.0
23005.0  vs  22129.0
42065.0  vs  41925.0
31102.0  vs  30748.0
51244.0  vs  51042.0
7797.0  vs  7834.0
83148.0  vs  83689.0
78479.0  vs  78367.0
70817.0  vs  70978.0
108057.0  vs  108210.0
29589.0  vs  29374.0
74576.0  vs  74631.0
64845.0  vs  64591.0
62820.0  vs  62684.0
89088.0  vs  89021.0
115814.0  vs  115451.0
93935.0  vs  93668.0
96450.0  vs  96225.0
54041.0  vs  54169.0
75532.0  vs  75800.0
80518.0  vs  80520.0
101873.0  vs  101439.0
78786.0  vs  78713.0
35379.0  vs  34896.0
66755.0  vs  66815.0
65465.0  vs  65471.0
77211.0  vs  77055.0
98205.0  vs  97903.0
61194.0  vs  61218.0
87264.0  vs  87548.0
35661.0  vs  35032.0
70607.0  vs  70572.0
87004.0  vs  87533.0
34103.0  vs  33616.0
65358.0  vs  65498.0
16931.0  vs  18934.0
85589.0  vs  85813.0
41359.0  vs  41212.0
36689.0  vs  36604.0
84049.0  vs  83916.0
126061.0  vs  126067.0
78482.0  vs  78500.0
60443.0  vs  60223.0
14580.0  vs  14563.0
57547.0  vs  58023.0
30683.0  vs  30601.0
85222.0  vs  85225.0
95326.0  vs  95653.0
39984.0  vs  40100.0
99758.0  vs  99736.0
30387.0  vs  30107.0
61886.0  vs  61674.0
73997.0  vs  73943.0
60570.0  vs  60599.0
41518.0  vs  40291.0
65470.0  vs  65437.0
19272.0  vs  18866.0
41933.0  vs  42178.0
36459.0  vs  35972.0
66085.0  vs  66660.0
110796.0  vs  110372.0
64174.0  vs  64028.0
88006.0  vs  87733.0
49448.0  vs  49507.0
123092.0  vs  123353.0
68876.0  vs  68873.0
72029.0  vs  71839.0
53940.0  vs  53106.0
106141.0  vs  106345.0
124579.0  vs  124156.0
81986.0  vs  81960.0
78773.0  vs  79147.0
76846.0  vs  76700.0
46014.0  vs  45934.0
60788.0  vs  61153.0
64939.0  vs  64883.0
71199.0  vs  71435.0
96040.0  vs  95741.0
64152.0  vs  64463.0
96607.0  vs  96814.0
103554.0  vs  103166.0
79367.0  vs  79133.0
56398.0  vs  56684.0
105187.0  vs  104938.0
8802.0  vs  10073.0
62547.0  vs  62565.0
63319.0  vs  63228.0
34617.0  vs  34377.0
59691.0  vs  58809.0
91535.0  vs  91525.0
48771.0  vs  48543.0
22980.0  vs  22949.0
22399.0  vs  21247.0
28351.0  vs  28908.0
71646.0  vs  71723.0
40603.0  vs  40271.0
22090.0  vs  21682.0
109449.0  vs  109565.0
35156.0  vs  34862.0
14296.0  vs  14712.0
108703.0  vs  109022.0
74437.0  vs  74216.0
79016.0  vs  79029.0
57166.0  vs  57167.0
90220.0  vs  89919.0
61222.0  vs  61438.0
109932.0  vs  109507.0
99427.0  vs  99420.0
67336.0  vs  67531.0
112348.0  vs  111906.0
18252.0  vs  18307.0
66530.0  vs  67139.0
47030.0  vs  47121.0
68607.0  vs  68214.0
16689.0  vs  16105.0
87075.0  vs  86888.0
75971.0  vs  75825.0
69154.0  vs  69545.0
19680.0  vs  20127.0
33254.0  vs  33405.0
57068.0  vs  59340.0
39280.0  vs  39390.0
97112.0  vs  96838.0
85558.0  vs  85117.0
39984.0  vs  40358.0
38231.0  vs  38035.0
12332.0  vs  12831.0
103452.0  vs  103545.0
98537.0  vs  98038.0
61383.0  vs  62101.0
31348.0  vs  31120.0
48265.0  vs  48531.0
54776.0  vs  54938.0
13189.0  vs  13292.0
95332.0  vs  94976.0
38038.0  vs  37336.0
63298.0  vs  63172.0
74894.0  vs  74899.0
71688.0  vs  71437.0
77126.0  vs  77428.0
49960.0  vs  50048.0
23507.0  vs  23053.0
94391.0  vs  94393.0
100705.0  vs  100481.0
83772.0  vs  83476.0
44043.0  vs  43874.0
26406.0  vs  26096.0
64754.0  vs  64305.0
98898.0  vs  98822.0
45250.0  vs  44890.0
93092.0  vs  93009.0
74609.0  vs  74813.0
92117.0  vs  92230.0
25974.0  vs  25556.0
18920.0  vs  18751.0
27674.0  vs  27881.0
103283.0  vs  103119.0
33730.0  vs  33872.0
31995.0  vs  31728.0
73845.0  vs  73791.0
87003.0  vs  86742.0
50006.0  vs  49552.0
58908.0  vs  58842.0
37256.0  vs  37187.0
62702.0  vs  62679.0
96444.0  vs  95988.0
30610.0  vs  30495.0
54061.0  vs  54066.0
69449.0  vs  69697.0
25212.0  vs  24570.0
34387.0  vs  34278.0
34545.0  vs  34235.0
49687.0  vs  52599.0
67645.0  vs  67608.0
35171.0  vs  35806.0
74908.0  vs  74246.0
19036.0  vs  18791.0
33105.0  vs  32304.0
106848.0  vs  106815.0
61613.0  vs  61590.0
57794.0  vs  59869.0
43301.0  vs  43062.0
96217.0  vs  95916.0
72589.0  vs  72445.0
35253.0  vs  35978.0
83423.0  vs  83382.0
72290.0  vs  72219.0
63151.0  vs  63180.0
72152.0  vs  71580.0
60803.0  vs  60907.0
20798.0  vs  21238.0
30751.0  vs  27151.0
93118.0  vs  93033.0
100337.0  vs  99982.0
95648.0  vs  95802.0
76633.0  vs  76526.0
13873.0  vs  13780.0
84802.0  vs  84869.0
101104.0  vs  101322.0
19520.0  vs  19184.0
59308.0  vs  59285.0
84860.0  vs  84630.0
58844.0  vs  58483.0
55201.0  vs  55464.0
92431.0  vs  92439.0
118368.0  vs  118366.0
97460.0  vs  97332.0
43803.0  vs  43192.0
51045.0  vs  50820.0
70839.0  vs  70774.0
86889.0  vs  86973.0
80110.0  vs  79924.0
63116.0  vs  63688.0
22481.0  vs  22472.0
50849.0  vs  50403.0
67235.0  vs  66888.0
57112.0  vs  56897.0
101053.0  vs  101520.0
46068.0  vs  46113.0
63509.0  vs  63622.0
40209.0  vs  39556.0
84862.0  vs  84706.0
104721.0  vs  103979.0
101674.0  vs  101240.0
79142.0  vs  78813.0
56466.0  vs  56246.0
100421.0  vs  100088.0
101556.0  vs  101496.0
110285.0  vs  110488.0
72979.0  vs  72937.0
82187.0  vs  81857.0
59928.0  vs  59772.0
71598.0  vs  71412.0
111915.0  vs  111676.0
72690.0  vs  72582.0
26315.0  vs  26055.0
59279.0  vs  59839.0
43177.0  vs  43424.0
62026.0  vs  61762.0
17900.0  vs  17555.0
90674.0  vs  90243.0
67577.0  vs  67370.0
6416.0  vs  6355.0
30686.0  vs  30468.0
65804.0  vs  66437.0
103660.0  vs  103600.0
69338.0  vs  69260.0
67454.0  vs  67546.0
73421.0  vs  73454.0
57216.0  vs  57265.0
109341.0  vs  109075.0
81842.0  vs  81738.0
53734.0  vs  53147.0
79361.0  vs  79759.0
37360.0  vs  37780.0
61803.0  vs  62222.0
49944.0  vs  50048.0
98872.0  vs  98529.0
53156.0  vs  53537.0
4926.0  vs  4764.0
39868.0  vs  39062.0
8542.0  vs  9090.0
44148.0  vs  44309.0
82723.0  vs  82373.0
46251.0  vs  46099.0
98449.0  vs  98487.0
79707.0  vs  79473.0
112036.0  vs  112100.0
72070.0  vs  71690.0
108786.0  vs  108289.0
62155.0  vs  62168.0
71913.0  vs  71100.0
64097.0  vs  63965.0
30915.0  vs  30384.0
47960.0  vs  48339.0
48352.0  vs  48396.0
102248.0  vs  101129.0
106278.0  vs  106498.0
55657.0  vs  55887.0
104415.0  vs  104365.0
71665.0  vs  71714.0
78081.0  vs  78109.0
29025.0  vs  29213.0
121352.0  vs  121059.0
71886.0  vs  71768.0
22370.0  vs  22563.0
85524.0  vs  85362.0
43797.0  vs  43742.0
26300.0  vs  26638.0
23108.0  vs  23318.0
86354.0  vs  86338.0
42054.0  vs  41541.0
23677.0  vs  24027.0
73201.0  vs  73098.0
89020.0  vs  88724.0
56304.0  vs  56423.0
47476.0  vs  47470.0
75495.0  vs  75428.0
8126.0  vs  8083.0
120856.0  vs  121244.0
65126.0  vs  65062.0
77974.0  vs  78132.0
54303.0  vs  54084.0
101009.0  vs  100624.0
62267.0  vs  64069.0
17435.0  vs  17222.0
80248.0  vs  80789.0
75114.0  vs  74881.0
48406.0  vs  48695.0
62916.0  vs  62800.0
40332.0  vs  40341.0
50727.0  vs  50466.0
69250.0  vs  69779.0
 validation accuracy 0.00, validation perplexity 1.00

Process finished with exit code 0

'''