def p(path, npath):
    fn = open(npath, 'w')
    with open(path) as f:
        corpus = []
        for i in f.readlines():
            i = i.strip().split(',')[-1]
            fn.write(f'{i}\n')

p('person1.txt', 'person1-dailydialog-rest.txt')
p('person2.txt', 'person2-dailydialog-rest.txt')
p('person3.txt', 'person3-dailydialog-rest.txt')
