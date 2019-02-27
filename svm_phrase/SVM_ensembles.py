import numpy as np
from sklearn.svm import LinearSVC
import sklearn


class Classifiers:
    """
    Usage:
           1. Generating models:
           After creating instances of this class, invoke the method 'training'
           with the path as its parameter, and after that, when you want to get
           the resulting model, just invoke its class variable 'model'

           2. Test model:
           Invoking 'testing' function with path to the test set as parameter,
           it will return f1 score
    """
    def __init__(self, n_value, intercept, mode='character'):
        self.n_value = n_value
        self.mode = mode
        if self.mode != 'character' and self.mode != 'word':
            raise ValueError('the mode has to be either character or word')
        self.features = None
        self.dialects = list()
        self.n_gram_set = set()
        self.length = 0
        self.width = 0
        self.model = None
        self.intercept = intercept

    def _char_n_grams(self, sentence):
        return [sentence[i: i + self.n_value] for i in range(len(sentence) - self.n_value + 1)]

    def _word_n_grams(self, sentence):
        ngram = list()
        sentence = sentence.strip().split(" ")
        for i in range(len(sentence) - self.n_value + 1):
            gram = ""
            for j in range(self.n_value):
                gram = gram + sentence[i + j]
            ngram.append(gram)
        return ngram

    def training(self, training_set_path):
        sentences = list()

        with open(training_set_path, 'r', encoding='utf8') as training_file:
            for line in training_file:
                the_sentence, the_dialect = line.strip().split('\t')
                sentences.append('#' + the_sentence + '#')
                self.dialects.append(the_dialect)

        sentence_features = list()

        if self.mode == 'character':
            for each in sentences:
                ngram = self._char_n_grams(each)
                sentence_features.append(set(ngram))
                self.n_gram_set.update(ngram)
        elif self.mode == 'word':
            for each in sentences:
                ngram = self._word_n_grams(each)
                sentence_features.append(set(ngram))
                self.n_gram_set.update(ngram)

        self.length = len(sentences)
        self.width = len(self.n_gram_set)
        self.features = np.zeros((self.length, self.width), dtype=np.int8)

        for i, sent in enumerate(sentence_features):
            for j, ngram in enumerate(self.n_gram_set):
                if ngram in sent:
                    self.features[i, j] = 1

        self._svc()

    def _svc(self):
        self.model = LinearSVC()
        self.model.fit(self.features, self.dialects)

    def testing(self, testing_set_path):
        test_sentences = []
        test_dialects = []
        with open(testing_set_path, 'r', encoding='utf8') as test_file:
            for line in test_file:
                s, label = line.strip().split('\t')
                test_sentences.append('#' + s + '#')
                test_dialects.append(label)

        s_feat = []

        if self.mode == 'character':
            for s in test_sentences:
                ngram = self._char_n_grams(s)
                s_feat.append(set(ngram))
        elif self.mode == 'word':
            for s in test_sentences:
                ngram = self._word_n_grams(s)
                s_feat.append(set(ngram))

        test_features = np.zeros((self.length, self.width), dtype=np.int8)

        for i, s in enumerate(s_feat):
            for j, ngram in enumerate(self.n_gram_set):
                if ngram in s:
                    test_features[i, j] = 1

        for i in range(self.length-len(test_sentences)):
            test_dialects.append(self.intercept)

        result = self.model.predict(X=test_features)
        f1_score_T = sklearn.metrics.f1_score(test_dialects, result[:len(test_dialects)], pos_label='T', average='binary')
        f1_score_M = sklearn.metrics.f1_score(test_dialects, result[:len(test_dialects)], pos_label='M', average='binary')
        score = self.model.score(test_features, test_dialects)
        return (f1_score_T + f1_score_M) / 2
