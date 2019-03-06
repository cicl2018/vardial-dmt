simp = []
with open('../results/simp.label', 'r')as f1:
    for each_line in f1:
        simp.append(each_line)

trad = []
with open('../results/trad.label', 'r')as f2:
    for each_line in f2:
        trad.append(each_line)


same = 0
diff = 0

for i, each in enumerate(simp):
    if trad[i] == each:
        same += 1
    else:
        diff += 1

print('labels are the same:', same, 'labels are different:', diff)
