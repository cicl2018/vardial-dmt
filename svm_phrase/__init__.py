from svm_phrase import SVM_ensembles
import datetime
from multiprocessing import Pool

parameter = [[1, 'T', 'character'], [2, 'T', 'character'], [3, 'T', 'character'], [4, 'T', 'character'],
             [1, 'T', 'word'], [2, 'T', 'word'], [3, 'T', 'word'], [4, 'T', 'word']]


def run(parameter_list):
    a_classifier = SVM_ensembles.Classifiers(parameter_list[0], parameter_list[1], parameter_list[2])
    a_classifier.training('../kars/trad_train_3000.txt')
    parameter_list.append(a_classifier.testing('../data/simp_2000.test'))
    return parameter_list


if __name__ == '__main__':
    start = datetime.datetime.now()
    start = ('%d-%d-%d-%d:%d' % (start.year, start.month, start.day, start.hour, start.minute))
    print(start)

    p = Pool(4)
    results = p.map(run, parameter)
    print(results)

    end = datetime.datetime.now()
    end = ('%d-%d-%d-%d:%d' % (end.year, end.month, end.day, end.hour, end.minute))
    print(end)
