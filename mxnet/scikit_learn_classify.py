'''
use scikit-learn to do classification of machine learning.

Training data is movie rates: some text comments and the label with is(1 is positive and 0 is negtive).
We convert the text comment list into a matrix by  tfidf algorithm,
each line is for a comment, each column in a line is the score of this comment to one word.
'''
import d2l
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import random
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import sklearn.tree
import sklearn.neighbors
import sklearn.linear_model
from sklearn.metrics import confusion_matrix
import scipy.sparse.csr



compute = False

if compute:
    #d2l.download_imdb('~/.mxnet/datasets/') #预定义的影评数据集
    data = d2l.read_imdb('train', '~/.mxnet/datasets/')
    data = [(data[0][i], data[1][i]) for i in range(len(data[0]))]
    random.shuffle(   data  )
    X = [ data[i][0] for i in range(len(data))]
    Y = [ data[i][1] for i in range(len(data))]

    cv = CountVectorizer()
    tfidf = TfidfTransformer()

    X = tfidf.fit_transform(cv.fit_transform(X)) # type:scipy.sparse.csr.csr_matrix
    words = cv.get_feature_names()



    print(words[-10:])

    with open("./data/naive_data.pkl", "wb") as f:
        pickle.dump((X, Y), f)
else:
    with open("./data/naive_data.pkl", "rb") as f:
        (X, Y) = pickle.load(f)

print(X.shape)
print(X.min())
print(X.max())

sep = int(X.shape[0] * 0.8)
train_X = X[0:sep]
test_X = X[sep:]
train_Y = Y[0:sep]
test_Y = Y[sep:]

#classifier = BernoulliNB()
classifier = sklearn.linear_model.LogisticRegression(verbose=1)
#classifier = sklearn.neighbors.KNeighborsClassifier()
#classifier = sklearn.tree.DecisionTreeClassifier()
classifier.fit(train_X, train_Y)

y = classifier.predict(test_X)
result = np.array([1 if y[i] == test_Y[i] else 0 for i in range(len(test_Y))])
print(result.sum() / result.size)

print(confusion_matrix(y, test_Y))





