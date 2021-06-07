def p(path, npath):
    fn = open(npath, 'w')
    with open(path) as f:
        corpus = []
        for i in f.readlines():
            i = i.strip().split(',')[-1]
            fn.write(f'{i}\n')

p('person1.txt', 'person1-cornell-rest.txt')
p('person2.txt', 'person2-cornell-rest.txt')
p('person3.txt', 'person3-cornell-rest.txt')
