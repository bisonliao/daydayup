'''
classify the text with :
1. bags of words representation.
2. tfidf representation, get a little higher accuracy

'''
import sklearn
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import os
import sklearn.utils
import scipy.sparse.csr
from sklearn.naive_bayes import MultinomialNB

#coding:utf8

# demo dataset from sklearn
twenty_train = fetch_20newsgroups(subset="train", data_home="e:/DeepLearning/data/nlp_data/corpus/20news/", download_if_missing=True) #type:sklearn.utils.Bunch
twenty_test = fetch_20newsgroups(subset="test",   data_home="e:/DeepLearning/data/nlp_data/corpus/20news/", download_if_missing=True)
print("train sample number:", len(twenty_train.data))
print("test sample number:", len(twenty_test.data))

############## bags of words ###################################
print("\nuse bags of words representation: ")
# bags of words:
# 1. Assign a ﬁxed integer id to each word occurring in any document of the training set (for instance by building a dictionary from words to integer indices).
# 2. For each document #i, count the number of occurrences of each word w and store it in X[i, j] as the value of feature #j where j is the index of word w in the dictionary.
count_vect = CountVectorizer()
count_vect.fit(twenty_train.data + twenty_test.data)
X_train_counts =count_vect.transform(twenty_train.data)  #type:scipy.sparse.csr.csr_matrix
X_test_counts = count_vect.transform(twenty_test.data)#type:scipy.sparse.csr.csr_matrix
# the value in X is word counts, integer.
print("features number:", len(count_vect.vocabulary_.keys()))

classifier = MultinomialNB()
classifier.fit(X_train_counts, twenty_train.target)
y = classifier.predict(X_train_counts)


def checkTopnAcc(pred:np.ndarray, label, k=1):
    result = list()
    if k == 1:
        for i in range(len(label)):
            if pred[i] == label[i]:
                result.append(1)
            else:
                result.append(0)
    else:
        for i in range(len(label)):
            index = np.argsort(pred[i])[-k:]
            if label[i] in index:
                result.append(1)
            else:
                result.append(0)
    return( np.array(result).sum()/len(result))

print(checkTopnAcc(y, twenty_train.target))# 92% accuracy

y = classifier.predict(X_test_counts)
print("top1:",checkTopnAcc(y, twenty_test.target)) # 76% accuracy

# check topN accuracy
y = classifier.predict_proba(X_test_counts)
print("top3:", checkTopnAcc(y, twenty_test.target, 3)) #88%

##########################   tfidf   ################################
print("\nuse tfidf representation: ")
tfidf = TfidfVectorizer()
tfidf.fit(twenty_train.data + twenty_test.data)
print("features number:", len(tfidf.vocabulary_.keys()))
X_train_tfidf =tfidf.transform(twenty_train.data)  #type:scipy.sparse.csr.csr_matrix
X_test_tfidf = tfidf.transform(twenty_test.data)#type:scipy.sparse.csr.csr_matrix
#the value in X is tfidf value, float < 1

classifier = MultinomialNB()
classifier.fit(X_train_tfidf, twenty_train.target)
y = classifier.predict(X_train_tfidf)
print(checkTopnAcc(y, twenty_train.target))# 93% accuracy

y = classifier.predict(X_test_tfidf)
print("top1:",checkTopnAcc(y, twenty_test.target)) # 77% accuracy

# check topN accuracy
y = classifier.predict_proba(X_test_tfidf)
print("top3:", checkTopnAcc(y, twenty_test.target, 3)) #90%
