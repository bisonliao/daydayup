'''
基于imdb的影评数据，训练一个RNN模型，可以判断一条影评是positive还是negtive的。
大量的函数都封装好了，在d2lzh包里。
'''
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
import mxnet
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils


batch_size = 64
embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, mxnet.gpu(0)
lr, num_epochs = 0.01, 5

####################################################################
# prepair dataset
#d2l.download_imdb('~/.mxnet/datasets/') #预定义的影评数据集
train_data = d2l.read_imdb('train', '~/.mxnet/datasets/')
test_data = d2l.read_imdb('test','~/.mxnet/datasets/')

vocab = d2l.get_vocab_imdb(train_data) # 分词，建立词汇表，过滤掉出现次数较少的词

train_set = gdata.ArrayDataset(*d2l.preprocess_imdb(train_data, vocab)) #  preprocess补齐长度，使其可以组成minibatch
test_set = gdata.ArrayDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
print('#batches:', len(train_iter) )


####################################################################
# define network
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经⽹络
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers, bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs的形状是(批量⼤⼩, 词数)，因为LSTM需要将序列作为第⼀维，所以将输⼊转置后 # 再提取词特征，输出形状为(词数, 批量⼤⼩, 词向量维度)
        embeddings = self.embedding(inputs.T)
        # rnn.LSTM只传⼊输⼊embeddings，因此只返回最后⼀层的隐藏层在各时间步的隐藏状态。 # outputs形状是(词数, 批量⼤⼩, 2 * 隐藏单元个数)
        outputs = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输⼊。它的形状为 # (批量⼤⼩, 4 * 隐藏单元个数)。
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs


net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)

# use pretrained word embedding for embedding layer
glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')

####################################################################
# train

trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

####################################################################
# use trained model
print(d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))
print(d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad']) )
