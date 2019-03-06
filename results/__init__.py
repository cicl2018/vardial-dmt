#from svm_phrase import SVM_ensembles
import SVM_ensembles
#from svm_phrase import fusion_methods
import fusion_methods
import datetime
from multiprocessing import Pool


#parameter = [[1, 'T', 'character'], [2, 'T', 'character'], [3, 'T', 'character'], [4, 'T', 'character'],
#             [1, 'T', 'word'], [2, 'T', 'word'], [3, 'T', 'word'], [4, 'T', 'word']]

parameter = [[4, 'T', 'character'], [3, 'T', 'character'], [2, 'T', 'character'], [5, 'T', 'character']]  # best combination


def run(parameter_list):
    a_classifier = SVM_ensembles.Classifiers(parameter_list[0], parameter_list[1], parameter_list[2])
    a_classifier.training('../final_data/dmt-simp-train.txt')
    parameter_list.append(a_classifier.testing('../final_data/dmt-simp-test.txt'))
    return parameter_list


if __name__ == '__main__':
    start = datetime.datetime.now()
    start = ('%d-%d-%d-%d:%d' % (start.year, start.month, start.day, start.hour, start.minute))
    print(start)
    p = Pool(8)
    results_list = p.map(run, parameter)
    #for each in results_list:
    #    print(each[0], each[2])
    #    print(each[3][0])
    #    print('')
    # for item in results_list:
    #   print(item)

    final_f1_score = fusion_methods.mean_probability_rule_fusion(results_list, len(parameter))
    print(final_f1_score)
    end = datetime.datetime.now()
    end = ('%d-%d-%d-%d:%d' % (end.year, end.month, end.day, end.hour, end.minute))
    print(end)

