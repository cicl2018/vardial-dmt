#!/usr/bin/python3

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# phrase-ngram
def ngrams(s, n):
    ngram = []
    s = s.strip().split(" ")
    for i in range(len(s) - n + 1):
        gram = ""
        for j in range(n):
            gram = gram + s[i+j]
        ngram.append(gram)
    return ngram


if __name__ == '__main__':

    t0 = time.time()
    print('start: {}'.format(t0))

    mainland = {}
    taiwan = {}

    words = []
    dialect = []

    with open('blank.txt', 'r') as fp:
        for index, line in enumerate(fp):
            w, label = line.strip().split('\t')
            words.append(w)
            dialect.append(label)

    ng_set = set()
    w_feat = []

    for w in words:
        print(w)
        bg = ngrams(w, 3)
        print(bg)
        w_feat.append(set(bg))
        ng_set.update(bg)

    ng_set = sorted(ng_set)

    features = np.zeros((len(words), len(ng_set)), dtype=np.int8)

    for i, w in enumerate(w_feat):
        for j, bg in enumerate(ng_set):
            if bg in w:
                features[i, j] = 1

    features.sum(axis=1)

    regression = LogisticRegression()
    regression.fit(features, dialect)
    print('Regression score: {}'.format(regression.score(features, dialect)))

    svm = LinearSVC()
    svm.fit(features, dialect)
    print('SVM score: {}'.format(svm.score(features, dialect)))

    t1 = time.time() - t0
    print('\nfinished: {} seconds'.format(t1))
