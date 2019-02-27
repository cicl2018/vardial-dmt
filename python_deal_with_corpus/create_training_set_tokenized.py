import os

file_add = '/home/nianheng/Documents/dmt/tokenized_simplified/'

train_list = list()

for file in os.listdir(file_add):
    if file.endswith('.txt'):
        train_list.append(file)


for each_file in train_list:
    count = 0
    with open(file_add+each_file, 'r', encoding='utf8')as the_file:
        for lines in the_file:
            lines = lines.strip()
            if lines.endswith('T') and count < 16770:
                with open(file_add+each_file[:-6]+'_guoyu.train', 'a+', encoding='utf8')as t_file:
                    t_file.write(lines[:-1]+'\n')
            elif lines.endswith('M') and count < 16770:
                with open(file_add + each_file[:-6] + '_mandarin.train', 'a+', encoding='utf8')as t_file:
                    t_file.write(lines[:-1]+'\n')
            elif lines.endswith('T') and count >= 16770:
                with open(file_add + each_file[:-6] + '_guoyu.test', 'a+', encoding='utf8')as t_file:
                    t_file.write(lines[:-1]+'\n')
            elif lines.endswith('M') and count >= 16770:
                with open(file_add + each_file[:-6] + '_mandarin.test', 'a+', encoding='utf8')as t_file:
                    t_file.write(lines[:-1]+'\n')
            count += 1
