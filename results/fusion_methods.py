import numpy as np
import sklearn
#from svm_phrase import  Yule_coefficient
import Yule_coefficient


#def test_dialects_func(testing_set_path):
#    test_dialects = list()
#    with open(testing_set_path, 'r', encoding='utf8') as test_file:
#        for line in test_file:
#            s, label = line.strip().split('\t')
#            test_dialects.append(label)
#    return test_dialects


def mean_probability_rule(test_features, clf):
    proba = clf.predict_proba(test_features)
    classes = clf.classes_
    return proba, classes


def mean_probability_rule_fusion(result, n):
    #test_dialects = test_dialects_func(test_path)
    #Yule_coefficient.yule_co_pairwise(result, len(test_dialects))

    shape = np.shape(result[0][3][0])
    sentences_m_prob = np.zeros((shape[0], shape[1]), dtype=np.int8)
    result_label = list()
    for each_classifier in result:
        prob_matrix = each_classifier[3][0]
        prob_label = each_classifier[3][1]
        print(prob_label)
        if prob_label[0] != 'M':
            print('wrong label')
        else:
            sentences_m_prob = np.add(prob_matrix, sentences_m_prob)
    result_matrix = np.true_divide(sentences_m_prob, n)
    for each_row in result_matrix:
        if each_row[0] > each_row[1]:
            result_label.append('M')
        else:
            result_label.append('T')

    with open('simp.label', 'a+', encoding='utf8')as out:
        for i, each in enumerate(result_label[:2000]):
            out.write(each+'\n')

    #f1_score = sklearn.metrics.f1_score(test_dialects, result_label[:len(test_dialects)], average='macro')
    #return f1_score


