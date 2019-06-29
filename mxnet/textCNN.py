'''
基于imdb的影评数据，训练一个CNN模型，可以判断一条影评是positive还是negtive的。
大量的函数都封装好了，在d2lzh包里。
CNN的设计应该是有点讲究的，不是简单的层级叠加卷积层就可以。
'''
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
import mxnet
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import pickle


batch_size = 64
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = mxnet.gpu(0)
lr, num_epochs = 0.001, 5

####################################################################
# prepair dataset
#d2l.download_imdb('~/.mxnet/datasets/') #预定义的影评数据集
train_data = d2l.read_imdb('train', '~/.mxnet/datasets/')
test_data = d2l.read_imdb('test','~/.mxnet/datasets/')

vocab = d2l.get_vocab_imdb(train_data) # 分词，建立词汇表，过滤掉出现次数较少的词

train_set = gdata.ArrayDataset(*d2l.preprocess_imdb(train_data, vocab)) #  preprocess补齐长度为500个词，使其可以组成minibatch
test_set = gdata.ArrayDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
print('#batches:', len(train_iter) )


####################################################################
# define network
class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab),    embed_size)
        # 不参与训练的嵌⼊层
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # 时序最⼤池化层没有权重，所以可以共⽤⼀个实例
        self.pool = nn.GlobalMaxPool1D()
        self.convs = nn.Sequential()
        # 创建多个⼀维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # 将两个形状是(批量⼤⼩, 词数, 词向量维度)的嵌⼊层的输出按词向量连结
        embeddings = nd.concat( self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # 根据Conv1D要求的输⼊格式，将词向量维，即⼀维卷积层的通道维，变换到前⼀维
        embeddings = embeddings.transpose((0, 2, 1))
        # 对于每个⼀维卷积层，在时序最⼤池化后会得到⼀个形状为(批量⼤⼩, 通道⼤⼩, 1)的
        # NDArray。使⽤flatten函数去掉最后⼀维，然后在通道维上连结
        encoding = nd.concat(*[nd.flatten( self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # 应⽤丢弃法后使⽤全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)

# use pretrained word embedding for embedding layer
#怎么保证同一个词，例如beautiful在前面的vocab里的index和在glove里的index是相同的呢？
# 看起来和create函数的第三个参数很有关系, 从glove的大表和大矩阵里抽取一些行，形成一个子集的词汇表和小矩阵
# 官方文档如此解释该参数的意义：
#vocabulary (Vocabulary, default None) – It contains the tokens to index. Each indexed token will be associated with the loaded embedding vectors,
# such as loaded from a pre-trained token embedding file. If None, all the tokens from the loaded embedding vectors, such as loaded from a
# pre-trained token embedding file, will be indexed.
glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.collect_params().setattr('grad_req', 'null')

print('vocabulary size:', len(vocab))
print("beautiful index:", glove_embedding.token_to_idx['beautiful'], " ", vocab.to_indices('beautiful'))

####################################################################
# train

trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
net.save_parameters("./data/sentiment.param")
#net.load_parameters("./data/sentiment.param")

####################################################################
# use trained model
print(d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))
print(d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad']) )

labels = ["negative", "positive"]
right = 0
for X, y in test_iter:
    for i in range(batch_size):
        indices = [int(a) for a in X[i].asnumpy()]
        sentence = vocab.to_tokens(indices)

        yy = d2l.predict_sentiment(net, vocab, sentence)
        label = labels[ int(y[i].asscalar())]
        print(">>", yy, " vs ", label)
        if yy == label:
            right = right+1
        else:
            print(*sentence)
    break
print(right * 1.0 /batch_size)
