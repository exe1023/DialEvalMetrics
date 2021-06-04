#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.7.10


from bert_serving.client import BertClient
import numpy as np


class BERT_RUBER_refer():
    
    def __init__(self, test=False):
        if test:
            self.bc = BertClient()
        else:
            print('[!] Training and testing do not need the bert-as-service')
    
    def encode_sentence(self, sent):
        return self.bc.encode([sent]).squeeze(0)    # [3072]
    
    def cos_similarity(self, groundtruth, generated):
        if generated and groundtruth:
            gr = self.encode_sentence(groundtruth)
            ge = self.encode_sentence(generated)
            sim = np.dot(gr, ge) / (np.linalg.norm(gr) * np.linalg.norm(ge))
        else:
            sim = 0.0
        return sim
        

if __name__ == "__main__":
    refer = BERT_RUBER_refer()
    sim = refer.cos_similarity('大大大', '你 是 谁 啊')
    print(sim)