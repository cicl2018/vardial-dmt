#!/usr/bin/python3

import numpy as np
from sklearn.svm import LinearSVC
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def ngrams(s, n):
    return [s[i: i+n] for i in range (len(s) - n+1)]


def svc(features, label):
    m2 = LinearSVC()
    m2.fit(features, label)
    # print(m2.predict(X=test_matrix))
    return m2


def run_train(path):
    sentences = []
    dialect = []

    with open(path, 'r') as fp:
        for line in fp:
            s, label = line.strip().split('\t')
            sentences.append('#' + s + '#')
            dialect.append(label)

    ng_set = set()
    s_feat = []

    for s in sentences:
        bg = ngrams(s, 2)
        s_feat.append(set(bg))
        ng_set.update(bg)

    features = np.zeros((len(sentences), len(ng_set)), dtype=np.int8)

    for i, w in enumerate(s_feat):
        for j, bg in enumerate(ng_set):
            if bg in w:
                features[i, j] = 1

    return features, dialect, len(sentences), len(ng_set), ng_set


def run_test(path, len_sentences, len_ngrams, ng_set1):
    sentences = []
    dialect = []
    with open(path, 'r') as fp:
        for line in fp:
            w, g = line.strip().split('\t')
            sentences.append('#' + w + '#')
            dialect.append(g)

    s_feat = []

    for s in sentences:
        bg = ngrams(s, 2)
        s_feat.append(set(bg))

    features = np.zeros((len_sentences, len_ngrams), dtype=np.int8)

    for i, s in enumerate(s_feat):
        for j, bg in enumerate(ng_set1):
            if bg in s:
                features[i, j] = 1

    for i in range(1600):
        dialect.append('T')

    return features, dialect

if __name__ == '__main__':

    train = 'data/simp2000.train'
    test = 'data/simp400.test'

    features, dialect, len_sentences, len_ngrams, ng_set = run_train(train)
    test_matrix = np.zeros((len_sentences, len_ngrams), dtype=np.int8)
    svm_model = svc(features, dialect)
    print(svm_model.score(features, dialect))

    features_test, gender_test = run_test(test, len_sentences, len_ngrams, ng_set)
    print(svm_model.score(features_test, gender_test))
    print(svm_model.predict(features_test)[:10])
    # confusion_matrix(gender_test, pred_test)
