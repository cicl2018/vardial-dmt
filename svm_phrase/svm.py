#!/usr/bin/python3

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def char_ngrams(s, n):
	return [s[i: i + n] for i in range(len(s) - n + 1)]


def ngrams(s, n):
	ngram = []
	s = s.strip().split(" ")
	for i in range(len(s) - n + 1):
		gram = ""
		for j in range(n):
			gram = gram + s[i + j]
		ngram.append(gram)
	return ngram


# def logi_re(features, gender):
#     m = LogisticRegression()
#     m.fit(features, gender)
#     #print(m.predict(X=test_matrix))
#     return m


def svc_re(features, gender):
	m2 = LinearSVC()
	m2.fit(features, gender)
	# print(m2.predict(X=test_matrix))
	return m2


def run_train(path, n):
	sentences = []
	dialects = []
	with open(path, 'r', encoding='utf8') as fp:
		for line in fp:
			s, label = line.strip().split('\t')
			sentences.append('#' + s + '#')
			dialects.append(label)

	ng_set = set()
	s_feat = []

	for s in sentences:
		# ngram = ngrams(s, n)
		ngram = char_ngrams(s, n)
		s_feat.append(set(ngram))
		ng_set.update(ngram)

	# print(len(ng_set))
	# print(sorted(s_feat)[:5])

	features = np.zeros((len(sentences), len(ng_set)), dtype=np.int8)

	for i, s in enumerate(s_feat):
		for j, ngram in enumerate(ng_set):
			if ngram in s:
				features[i, j] = 1

	return features, dialects, len(sentences), len(ng_set), ng_set


def run_test(path, len1, len2, ng_set1, n):
	sentences = []
	dialects = []
	with open(path, 'r') as fp:
		for line in fp:
			s, label = line.strip().split('\t')
			sentences.append('#' + s + '#')
			dialects.append(label)

	s_feat = []

	for s in sentences:
		# ngram = ngrams(s, n)
		ngram = char_ngrams(s, n)
		s_feat.append(set(ngram))

	features = np.zeros((len1, len2), dtype=np.int8)

	for i, s in enumerate(s_feat):
		for j, ngram in enumerate(ng_set1):
			if ngram in s:
				features[i, j] = 1

	for i in range(1000):
		dialects.append('T')

	return features, dialects


if __name__ == '__main__':
	train = '../kars/trad_train_3000.txt'
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
	accuracy = ((3000 * score - 1000) / 2000) * 100
	print("TEST: ", accuracy, '%')
# pre_test = m.predict(features_test, gender_test)
# confusion_matrix(gender_test, pred_test)
# compare model with some base line
