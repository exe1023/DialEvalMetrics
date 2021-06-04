#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.3.28


'''
This file hybird the unreference score and reference score
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from rouge import Rouge

import argparse
import os
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm
import scipy
from scipy.stats.mstats import gmean
from scipy.stats import pearsonr, spearmanr
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

from reference_score import *
from unreference_score import *
from utils import *
import json

parser = argparse.ArgumentParser(description='RUBER utils script')
parser.add_argument('--pretrain', type=str, default='dailydialog')
parser.add_argument('--dataset', type=str, default='xiaohuangji', 
                    help='the dataset we used')
args = parser.parse_args()


def tokenizer(text, max_length=128):
    # use spacy tokenizer function
    return [tok for tok in text.split()[:max_length]]


class RUBER:
    
    def __init__(self):
        with open(f'data/{args.pretrain}/tgt-vocab.pkl', 'rb') as f:
            TARGET = pickle.load(f)
        with open(f'data/{args.pretrain}/src-vocab.pkl', 'rb') as f:
            srcv = pickle.load(f)
        with open(f'data/{args.pretrain}/tgt-vocab.pkl', 'rb') as f:
            tgtv = pickle.load(f)
        self.refer = RUBER_refer('./embedding/word_embedding.txt', TARGET,
                            pooling_type='max_min', dataset=args.pretrain)
        self.unrefer = RUBER_unrefer(srcv.get_vocab_size(), tgtv.get_vocab_size(),
                                     100, 100)
        self.tgtv = tgtv
        self.srcv = srcv
        get_best_model(args.pretrain, self.unrefer)
        
        if torch.cuda.is_available():
            self.unrefer.cuda()
            self.unrefer.eval()
        
    def process_input(self, msg, vocab):
        words = torch.LongTensor([vocab.get_index(i.lower()) for i in ['<sos>'] + tokenizer(msg) + ['<eos>']]).unsqueeze(0).transpose(0, 1)
        lengths = torch.tensor([torch.tensor(words.shape[0])])
        # [T, 1], [1]
        return words, lengths
    
    def normalize(self, scores):
        smin = min(scores)
        smax = max(scores)
        diff = smax - smin
        ret = [(s - smin) / diff for s in scores]
        return ret
        
    def score(self, query, groundtruth, reply, method='Min'):
        q, l = self.process_input(query, self.srcv)
        r, rl = self.process_input(reply, self.tgtv)
        
        if torch.cuda.is_available():
            q, l, r, rl = q.cuda(), l.cuda(), r.cuda(), rl.cuda()
            
        unrefer_score = self.unrefer(q, l, r, rl)    # [0, 1]
        # print(unrefer_score)
        unrefer_score = unrefer_score[0].item()
        # refer_score = self.normalize(refer_score)
        refer_score = self.refer.score(groundtruth, reply)   # [-1, 1]
        # refer_score = self.normalize(refer_score)
        
        # all in [0, 1]
        return unrefer_score, refer_score
    
    def scores(self, contexts, gs, rs, method='Min'):
        # batch process
        refer, unrefer = [], []
        for c, g, r in tqdm(zip(contexts, gs, rs)):
            unrefer_score, refer_score = self.score(c, g, r, method=method)
            refer.append(refer_score)
            unrefer.append(unrefer_score)
        refer = self.normalize(refer)
        unrefer = self.normalize(unrefer)
        ruber = self.hybird_score(refer, unrefer)
        
        return refer, unrefer, ruber
        
        
    def hybird_score(self, refer, unrefer, method='Min'):
        # make sure refer and unrefer has been normed
        if method == 'Min':
            return [min(a,b) for a,b in zip(refer, unrefer)]
        elif method == 'Max':
            return [max(a,b) for a,b in zip(refer, unrefer)]
        else:
            raise Exception("Can not find the right method")
        
        

def collection_result(contextp, groundp, predp):
    # context, groundtruth, generate
    context, groundtruth, reply = [], [], []
    with open(contextp) as f:
        for line in f.readlines():
            context.append(line.strip())
    with open(groundp) as f:
        for line in f.readlines():
            groundtruth.append(line.strip())
    with open(predp) as f:
        for line in f.readlines():
            reply.append(line.strip())
    return context, groundtruth, reply


def cal_BLEU(refer, candidate, ngram=1):
    smoothie = SmoothingFunction().method4
    if ngram == 1:
        weight = (1, 0, 0, 0)
    elif ngram == 2:
        weight = (0.5, 0.5, 0, 0)
    elif ngram == 3:
        weight = (0.33, 0.33, 0.33, 0)
    elif ngram == 4:
        weight = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu(refer, candidate, weights=weight, smoothing_function=smoothie)

def cal_ROUGE(refer, candidate):
    if not candidate:
        candidate = 'unk'
    rouge = Rouge()
    scores = rouge.get_scores(' '.join(candidate), ' '.join(refer))
    return scores[0]['rouge-2']['f']
    

def show(scores, model_scores, mode):
    print(f'========== Method {mode} result ==========')
    p, pp = pearsonr(scores, model_scores)
    p, pp = round(p, 5), round(pp, 5)
    s, ss = spearmanr(scores, model_scores)
    s, ss = round(s, 5), round(ss, 5)
    print('Pearson(p-value):', f'{p}({pp})')
    print('Spearman(p-value):', f'{s}({ss})')
    print(f'========== Method {mode} result ==========')
    return p, pp, s, ss
    

def read_human_score(path1, path2, path3):
    def read_file(path):
        with open(path) as f:
            score = []
            for line in f.readlines():
                score.append(float(line.strip()))
        return score
    score1 = read_file(path1)
    score2 = read_file(path2)
    score3 = read_file(path3)
    return score1, score2, score3
    


if __name__ == "__main__":
    random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
        
    model = RUBER()
    context, groundtruth, reply = collection_result(f'./data/{args.dataset}/context.txt',
                                                    f'./data/{args.dataset}/reference.txt',
                                                    f'./data/{args.dataset}/hypothesis.txt')
    print(f'[!] read file')
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    rouge2_scores = []
    
    # RUBER
    refers, unrefer, ruber = model.scores(context, groundtruth, reply, method='Min')

    with open(f'data/{args.dataset}/refer_score_{args.pretrain}.json', 'w') as f:
        json.dump(refers, f)
    
    with open(f'data/{args.dataset}/unrefer_score_{args.pretrain}.json', 'w') as f:
        json.dump(unrefer, f)

    with open(f'data/{args.dataset}/ruber_score_{args.pretrain}.json', 'w') as f:
        json.dump(ruber, f)


    # BLEU
    '''for c, g, r in zip(context, groundtruth, reply):
        if not g:
            g = '<unk>'
        # nlgscores = compute_individual_metrics([g], r, no_overlap=False,
        #                                        no_skipthoughts=True,
        #                                        no_glove=True)
        # cider_scores.append(nlgscores['CIDEr'])
        # meteor_scores.append(nlgscores['METEOR'])
        
        refer, condidate = g.split(), r.split()
        if not refer:
            refer = ['<unk>']
        if len(condidate) == 1:
            condidate.append('<unk>')
        bleu1_scores.append(cal_BLEU(refer, condidate, ngram=1))
        bleu2_scores.append(cal_BLEU(refer, condidate, ngram=2))
        bleu3_scores.append(cal_BLEU(refer, condidate, ngram=3))
        bleu4_scores.append(cal_BLEU(refer, condidate, ngram=4))
        rouge2_scores.append(cal_ROUGE(refer, condidate))
    print(f'[!] compute the score')
        
    # human scores
    h1, h2, h3 = read_human_score(f'./data/{args.dataset}/person1-{args.dataset}-rest.txt',
                                  f'./data/{args.dataset}/person2-{args.dataset}-rest.txt',
                                  f'./data/{args.dataset}/person3-{args.dataset}-rest.txt')
    print(f'[!] read human score')
    
    show(h1, h2, 'Human 1v2')
    show(h1, h3, 'Human 1v3')
    show(h2, h3, 'Human 2v3')
    show(h1, bleu1_scores, "BLEU-1")
    show(h1, bleu2_scores, "BLEU-2")
    show(h1, bleu3_scores, "BLEU-3")
    show(h1, bleu4_scores, "BLEU-4")
    show(h1, rouge2_scores, "ROUGE-2")
    su_p, su_pp, su_s, su_ss = show(h1, unrefer, "s_U")
    sr_p, sr_pp, sr_s, sr_ss = show(h1, refers, "s_R")
    u_p, u_pp, u_s, u_ss = show(h1, ruber, "RUBER")
    
    # rest into file
    with open(f'./data/{args.dataset}/result.txt', 'a') as f:
        f.write(f'su_p: {su_p}({su_pp}), su_s: {su_s}({su_ss})' + '\n')
        f.write(f'sr_p: {sr_p}({sr_pp}), sr_s: {sr_s}({sr_ss})' + '\n')
        f.write(f'u_p: {u_p}({u_pp}), u_s: {u_s}({u_ss})' + '\n')
    '''