
guoyu = '/home/nianheng/Documents/dmt/n_p_experiment/Test/tokenized_simplifi_guoyu.test'
mandarin = '/home/nianheng/Documents/dmt/n_p_experiment/Test/tokenized_simplifi_mandarin.test'
result = '/home/nianheng/Documents/dmt/n=8_p=3.87_+iteration/Test/result (copy).txt'

guoyu_set = set()
mandarin_set = set()

# should be - classified as
man_man_set = set()
man_guo_set = set()
guo_guo_set = set()
guo_man_set = set()

mandarin_label = 'mandarin'
guoyu_label = 'guoyu'

with open(guoyu, 'r')as guoyu_file:
    for lines in guoyu_file:
        lines = lines.strip()
        guoyu_set.add(lines)

with open(mandarin, 'r')as mandarin_file:
    for lines in mandarin_file:
        lines = lines.strip()
        mandarin_set.add(lines)

with open(result, 'r')as result_file:
    for lines in result_file:
        lines = lines.strip().split()
        if lines == []:
            break
        label = ''.join(lines[-1:])
        sentence = ' '.join(lines[:-1])
        if sentence in guoyu_set and label == guoyu_label:
            guo_guo_set.add(sentence)
        elif sentence in guoyu_set and label == mandarin_label:
            guo_man_set.add(sentence)
        elif sentence in mandarin_set and label == mandarin_label:
            man_man_set.add(sentence)
        elif sentence in mandarin_set and label == guoyu_label:
            man_guo_set.add(sentence)

accuracy = (len(guo_guo_set) + len(man_man_set))/(len(guo_guo_set) + len(man_man_set)+len(guo_man_set)+len(man_guo_set))
print('accuracy: '+str(accuracy))

mandarin_true_positive = len(man_man_set)
mandarin_false_positive = len(guo_man_set)
mandarin_false_negative = len(man_guo_set)
mandarin_true_negative = len(guo_guo_set)

mandarin_precision = mandarin_true_positive/(mandarin_true_positive+mandarin_false_positive)
mandarin_recall = mandarin_true_positive/(mandarin_true_positive+mandarin_false_negative)
mandarine_f1 = 2*mandarin_precision*mandarin_recall/(mandarin_precision+mandarin_recall)

print('mandarin f1: '+ str(mandarine_f1))

guoyu_true_positive = len(guo_guo_set)
guoyu_false_positive = len(man_guo_set)
guoyu_false_negative = len(guo_man_set)
guoyu_true_negative = len(man_man_set)

guoyu_precision = guoyu_true_positive/(guoyu_true_positive+guoyu_false_positive)
guoyu_recall = guoyu_true_positive/(guoyu_true_positive+guoyu_false_negative)
guoyu_f1 = 2*guoyu_precision*guoyu_recall/(guoyu_precision+guoyu_recall)

print('guoyu f1: '+str(guoyu_f1))

print('average f1: '+str((mandarine_f1+guoyu_f1)/2))
