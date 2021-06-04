#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.3.28


'''
This file is the model for automatic evaluation natural language reponse
refer to:
RUBER: An Unsupervised Method for Automatic Evaluation of Open-Domain Dialog Systems
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import random
import numpy as np
import pickle
import sys
from utils import *


class RUBER_refer():
    
    """Referenced Metric
    Measure the similarity between the groundtruth reply and generated reply
    use cosine score.
    Provide three pooling methods for generating sentence vector:
        [max_min | avg | all]
    default max_min pooling
    
    load the GloVe, according to source_vocab or target_vocab,
    construct the embedding matrix
    """
    
    def __init__(self, path, target_vocab, special_words=None, 
                 vector_size=300, pooling_type='max_min', dataset='tencent'):
        self.vector_size = vector_size
        self.special_words = special_words
        self.target_embed = pickle.load(open(f'./data/{dataset}/tgt-embed.pkl', 'rb'))
        
        if pooling_type=='max_min':
            self.pooling = self.max_min_pooling
        elif pooling_type=='avg':
            self.pooling = self.average_pooling
        else:
            self.pooling = self.all_pooling
    
    def find_vec(self, word):
        try:
            vec = self.target_embed[word]
        except:
            vec = self.target_embed['<unk>']
        return vec
    
    def max_min_pooling(self, sent):
        # max pool
        try:
            vectors = np.stack([self.find_vec(word) for word in sent])
        except:
            vectors = np.stack([self.find_vec(None), self.find_vec(None)])
        max_rest = np.max(vectors, axis=0)
        min_rest = np.min(vectors, axis=0)
        rest = np.concatenate([max_rest, min_rest])
        return rest
    
    def average_pooling(self, sent):
        if len(sent):
            svector = [self.find_vec(word) for word in sent]
        else:
            svector = [self.find_vec(None)]
        l = float(len(svector))
        return [sum([vec[i] for vec in svector]) / l for i in range(self.vector_size)]
        
    def all_pooling(self, sent):
        return np.concatenate((self.max_min_pooling(sent),
                               self.average_pooling(sent)), axis=0)    # 3 * self.vec_dim
    
    def score(self, groundtruth, generated):
        groundtruth = self.pooling(groundtruth.split())
        generated = self.pooling(generated.split())
        sim = np.dot(groundtruth, generated) / (np.linalg.norm(groundtruth) * np.linalg.norm(generated))
        if np.isnan(sim):
            sim = 0.0
        return sim


if __name__ == "__main__":
    # refer test
    with open('data/tgt-vocab.pkl', 'rb') as f:
        TARGET = pickle.load(f)
    refer = RUBER_refer('./embedding/word_embedding.txt', TARGET, pooling_type='avg')
    groundtruth = '好 漂亮'
    reply = '是不是 有什么 毛病'
    print(refer.score(groundtruth, reply))