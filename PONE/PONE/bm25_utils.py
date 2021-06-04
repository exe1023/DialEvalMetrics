from gensim.summarization import bm25
import random
import numpy as np
import jieba

class BM25Model:

    def __init__(self, corpus, lang='en', threshold=10000):
        # corpus: tgt-train.txt
        with open(corpus) as f:
            data = []
            for line in f.readlines():
                if lang == 'zh':
                    data.append(list(jieba.cut(line.strip())))
                else:
                    data.append(line.strip().split())
        ridx = random.sample(range(len(data)), threshold)
        # dict: give the key in (0, threshold) -> (0, 45000)
        self.dict = {idx: i for idx, i in enumerate(ridx)}
        data_ = [data[i] for i in ridx]
        self.model = bm25.BM25(data_)
        self.data = data
        print('[!] init the bm25 model over')

    def get_weighted(self, res):
        res = self.data[res]
        scores = self.model.get_scores(res)
        scores = np.array(scores)
        idx = np.argpartition(scores, -100)[-100:]
        idx = [self.dict[i] for i in idx if self.data[self.dict[i]] != res]
        if not idx:
            # idx is None
            return random.choice(range(len(self.data)))
        else:
            return random.choice(idx)

if __name__ == "__main__":
    pass
