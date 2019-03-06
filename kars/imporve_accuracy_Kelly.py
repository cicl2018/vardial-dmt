#when a model's accuracy start to decrease when the complexity increase, the model is overfitted
# coding: utf-8

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from collections import Counter


def ngrams(s, n):
    ngram = []
    # print("S", s)
    s = s.strip().split(" ")
    for i in range(len(s) - n + 1):
        gram = ""
        for j in range(n):
            gram = gram + s[i + j]
        ngram.append(gram)

    #print("ngram", ngram)
    return ngram


def logi_re(features, gender):
    m = LogisticRegression()
    m.fit(features, gender)
    #print(m.predict(X=test_matrix))
    return m


def svc_re(features, gender):
    m2 = LinearSVC()
    m2.fit(features, gender)
    #print(m2.predict(X=test_matrix))
    return m2


t_counts = Counter()  # counting taiwanese words frequency
m_counts = Counter()  # counting mandarin words frequency
total_counts = Counter()  # counting total number of words


def tm_count(word, gen):
    if gen == 'T':
        for w in word:
            t_counts[w] += 1
            total_counts[w] += 1
    elif gen == 'M':
        for w in word:
            m_counts[w] += 1
            total_counts[w] += 1


t_m_ratios = Counter()
t_m_raw_ratios = Counter()


def tf_idf():
    #print("LIST count:" + str(list(total_counts.most_common())))
    for word, cnt in list(total_counts.most_common()):
        if cnt > 1:
            t_m_ratio = t_counts[word] / float(m_counts[word] + 1)
            t_m_ratios[word] = t_m_ratio
            #print("word in count:"+ word)
            #print("ratio:" + str(t_m_ratio))

    t_m_raw_ratios = t_m_ratios
    #print("LIST:" + str(list(t_m_ratios.most_common(100))))
    for word, ratio in list(t_m_ratios.most_common()):
        t_m_ratios[word] = np.log(ratio + 0.01)
        #print("word: " + word)
        #print(t_m_ratios[word])
    list(reversed(t_m_ratios.most_common()))

def run_train(path, n):
    sentence = []
    dialects = []
    with open(path, 'r') as fp:
        for line in fp:
            w, g = line.strip().split('\t')
            # words.append('#'+w+'#')
            sentence.append('#' + w + '#')
            dialects.append(g)
            # tm_count(w, g)

    ng_set = set()
    w_feat = []

    for w, g in zip(sentence, dialects):
        bg = ngrams(w, n)
        tm_count(bg, g)
        tf_idf()
        w_feat.append(set(bg))
        ng_set.update(bg)

    features = np.zeros((len(sentence), len(ng_set)), dtype=np.int8)
    print("BG: " + str(bg))
    for i, w in enumerate(w_feat):
        for j, bg in enumerate(ng_set):
            if bg in w:
                features[i, j] = 1
                print("features old:")
                print(features[i,j])
                print(t_m_ratios[bg])
                #print(t_m_ratios["我的"])
                if t_m_ratios[bg] != 0:
                    features[i,j] = (float(t_m_ratios[bg])* float(features[i,j]))
                print("features new:")
                print(features[i, j])

    return features, dialects, len(sentence), len(ng_set), ng_set


def run_test(path, len1, len2, ng_set1, n):
    sentence = []
    dialects = []
    with open(path, 'r') as fp:
        for line in fp:
            w, g = line.strip().split('\t')
            sentence.append('#' + w + '#')
            dialects.append(g)

    w_feat = []

    for w in sentence:
        bg = ngrams(w, n)
        w_feat.append(set(bg))

    features = np.zeros((len1, len2), dtype=np.int8)


    for i, w in enumerate(w_feat):
        for j, bg in enumerate(ng_set1):
            if bg in w:
                features[i, j] = 1
                features[i, j] *= t_m_ratios[bg]
    for i in range(14770):
        dialects.append('T')

    return features, dialects


if __name__ == '__main__':
    train = '../data/simp_16770.train'
    test = '../data/simp_2000.test'
    features, dialect, len1, len2, ng_set = run_train(train, 3)
    test_matrix = np.zeros((len1, len2), dtype=np.int8)
    # m = logi_re(features, dialect)
    svm = svc_re(features, dialect)

    # print("TRAIN:", m.score(features, dialect))
    print("TRAIN:", svm.score(features, dialect))
    features_test, dialect_test = run_test(test, len1, len2, ng_set, 3)

    # print("TEST:", m.score(features_test, dialect_test))
    score = svm.score(features_test, dialect_test)
    accuracy = ((16770 * score - 14770) / 2000) * 100
    print("TEST: ", accuracy, '%')
    print("T", t_counts.most_common(20))
    print("M", m_counts.most_common(20))
    # pre_test = m.predict(features_test, gender_test)
    # confusion_matrix(gender_test, pred_test)
    # compare model with some base line
