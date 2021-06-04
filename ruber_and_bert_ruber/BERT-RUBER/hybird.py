#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.7.10


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from rouge import Rouge

import argparse
import pickle
import os
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm
import ipdb
import scipy
from scipy.stats.mstats import gmean
from scipy.stats import pearsonr, spearmanr
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

from reference_score import *
from unreference_score import *
from utils import *
from metric.metric import cal_BLEU
import json

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


def cal_BLEU_(refer, candidate, ngram=1):
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
    
    
def read_human_score(path1, path2):
    def read_file(path):
        with open(path) as f:
            score = []
            for line in f.readlines():
                score.append(float(line.strip()))
        return score
    score1 = read_file(path1)
    score2 = read_file(path2)
    return score1, score2


class BERT_RUBER:
    
    def __init__(self, dataset):
        self.refer = BERT_RUBER_refer()
        self.unrefer = BERT_RUBER_unrefer(768)
        
        #load_best_model(self.unrefer, dataset)
        load_best_model(self.unrefer, 'dailydialog')
        
        if torch.cuda.is_available():
            self.unrefer.cuda()
            self.unrefer.eval()
            
    def normalize(self, scores):
        smin = min(scores)
        smax = max(scores)
        diff = smax - smin
        ret = [(s - smin) / diff for s in scores]
        return ret
    
    def refer_score(self, refer, reply):
        return self.refer.cos_similarity(refer, reply)

    def unrefer_score(self, query, reply):
        #q = self.refer.encode_query(query)
        #r = self.refer.encode_sentence(reply)
        # g = self.refer.encode_sentence(groundtruth)
        # q, r, g = torch.from_numpy(q), torch.from_numpy(r), torch.from_numpy(g)
        #q, r = torch.from_numpy(q), torch.from_numpy(r)
        #q = q.unsqueeze(0)
        #r = r.unsqueeze(0)
        # g = g.unsqueeze(0)
        
        #if torch.cuda.is_available():
            # q, r, g = q.cuda(), r.cuda(), g.cuda()
        #    q, r = q.cuda(), r.cuda()
        q = self.refer.tokenizer([query], return_tensors='pt', padding=True, truncation=True)
        r = self.refer.tokenizer([reply], return_tensors='pt', padding=True, truncation=True)

        batch = {}
        for key, val in q.items():
            batch[f'q_{key}'] = val.view(1, 1, -1)

        for key, val in r.items():
            batch[f'r_{key}'] = val.view(1, 1, -1)
        _, unrefer_score = self.unrefer(batch)
        unrefer_score = unrefer_score[0].item()

        # refer_score = self.refer.cos_similarity(groundtruth, reply)
        
        # return unrefer_score, refer_score
        return unrefer_score

    def score_batch_unrefer(self, query, reply):
        q = self.refer.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        r = self.refer.tokenizer(reply, return_tensors='pt', padding=True, truncation=True)

        batch = {}
        for key, val in q.items():
            batch[f'q_{key}'] = val

        for key, val in r:
            batch[f'r_{key}'] = val
        unrefer_score = self.unrefer(batch)
        unrefer_score = unrefer_score[0].item()
        return unrefer_score.tolist()
    
    def scores(self, contexts, gs, rs, method='Min'):
        refer = []
        unrefer = []
        hybrid = []
        # for c, g, r in pbar:
        #     c = ''.join(c.split())
        #     g = ''.join(g.split())
        #     r = ''.join(r.split())
        #     if not r:
        #         # no words genereated
        #         r = '<unk>'
        #     if not c:
        #         c = '<unk>'
        #import ipdb; ipdb.set_trace()
        for idx in range(len(contexts)):
            c = contexts[idx]
            g = gs[idx]
            r = rs[idx]
            refer_score = self.refer_score(g, r)
            unrefer_score = self.unrefer_score(c, r)
            refer.append(refer_score)
            unrefer.append(unrefer_score)
        '''idx = 0
        batch = 512 
        while idx < len(contexts):
            c = contexts[idx:idx+batch]
            r = rs[idx:idx+batch]
            # unrefer_score, refer_score = self.score(c, g, r, method=method)
            unrefer_score = self.score_batch_unrefer(c, r)
            # refer.append(refer_score)
            unrefer.extend(unrefer_score)
            idx += batch
            # print(f'{idx} / {len(contexts)}', end='\r')
        '''
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
            
def obtain_test_data(path):
    with open(path) as f:
        context, groundtruth, pred = [], [], []
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            line = line[13:]
            if not line:
                line = '<unk>'
            if idx % 4 == 0:
                context.append(line)
            elif idx % 4 == 1:
                groundtruth.append(line)
            elif idx % 4 == 2:
                pred.append(line)
            else:
                pass
    return context, groundtruth, pred
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    if args.mode == 'experiment':
        model = BERT_RUBER(args.dataset)
        context, groundtruth, reply = collection_result(f'./data/{args.dataset}/src-test.txt',
                                                        f'./data/{args.dataset}/tgt-test.txt',
                                                        f'./data/{args.dataset}/pred.txt')
        print(f'[!] read file')
        bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
        rouge2_scores = []

        # BERT RUBER
        refers, unrefer, ruber = model.scores(context, groundtruth, reply, method='Min')
        # BLEU
        for c, g, r in zip(context, groundtruth, reply):
            refer, condidate = g.split(), r.split()
            bleu1_scores.append(cal_BLEU(refer, condidate, ngram=1))
            bleu2_scores.append(cal_BLEU(refer, condidate, ngram=2))
            bleu3_scores.append(cal_BLEU(refer, condidate, ngram=3))
            bleu4_scores.append(cal_BLEU(refer, condidate, ngram=4))
            rouge2_scores.append(cal_ROUGE(refer, condidate))
        print(f'[!] compute the score')

        # human scores
        h1, h2 = read_human_score('./data/lantian1-xiaohuangji-rest.txt',
                                  './data/lantian2-xiaohuangji-rest.txt')
        print(f'[!] read human score')

        show(h1, h2, 'Human')
        show(h1, bleu1_scores, "BLEU-1")
        show(h1, bleu2_scores, "BLEU-2")
        show(h1, bleu3_scores, "BLEU-3")
        show(h1, bleu4_scores, "BLEU-4")
        show(h1, rouge2_scores, "ROUGE-2")
        su_p, su_pp, su_s, su_ss = show(h1, unrefer, "BERT s_U")
        sr_p, sr_pp, sr_s, sr_ss = show(h1, refers, "BERT s_R")
        u_p, u_pp, u_s, u_ss = show(h1, ruber, "BERT RUBER")

        # rest into file
        with open(f'./data/{args.dataset}/result.txt', 'a') as f:
            f.write(f'su_p: {su_p}({su_pp}), su_s: {su_s}({su_ss})' + '\n')
            f.write(f'sr_p: {sr_p}({sr_pp}), sr_s: {sr_s}({sr_ss})' + '\n')
            f.write(f'u_p: {u_p}({u_pp}), u_s: {u_s}({u_ss})' + '\n')
    elif args.mode == 'generate':
        model = BERT_RUBER(args.dataset)
        #print(f'[!] ready to read data from ./data/{args.dataset}/{args.model}-pred.txt')
        #context, groundtruth, reply = obtain_test_data(f'./data/{args.dataset}/{args.model}-pred.txt')
        context, groundtruth, reply = collection_result(f'./data/{args.dataset}/context.txt',
                                                        f'./data/{args.dataset}/reference.txt',
                                                        f'./data/{args.dataset}/hypothesis.txt') 
        refers, unrefer, ruber = model.scores(context, groundtruth, reply, method='Min')
        # BERT RUBER
        with open(f'data/{args.dataset}/refer_score.json', 'w') as f:
            json.dump(refers, f)
    
        with open(f'data/{args.dataset}/unrefer_score.json', 'w') as f:
            json.dump(unrefer, f)

        with open(f'data/{args.dataset}/ruber_score.json', 'w') as f:
            json.dump(ruber, f)
        
        #unrefer = model.scores(context, groundtruth, reply, method='Min')
        # bleu = cal_BLEU(groundtruth, reply)
        # bleu1, bleu2, bleu3, bleu4 = round(bleu[0], 4), round(bleu[1], 4), round(bleu[2], 4), round(bleu[3], 4)
        # print(f'bleu(1/2/3/4): {bleu1}/{bleu2}/{bleu3}/{bleu4}')
        # bleu = sum([bleu1, bleu2, bleu3, bleu4]) / 4
        # print(f'BLEU-avg: {round(bleu, 4)}')
        
        # with open(f'./data/{args.dataset}/{args.model}-result.pkl', 'wb') as f:
        #     pickle.dump(unrefer, f)
        #     print(f'[!] write the file into ./data/{args.dataset}/{args.model}-result.pkl')
        f_unrefer = np.mean(unrefer)
        print(f'BERT-RUBER: {round(f_unrefer, 4)}')
    elif args.mode == 'bertscore':
        context, groundtruth, reply = obtain_test_data(f'./data/{args.dataset}/{args.model}-pred.txt')
        
        # add the BERTScore
        from bert_score import score
        _, _, bert_scores = score(reply, groundtruth, lang='en',
                                  rescale_with_baseline=True)
        bert_scores = bert_scores.tolist()
        bert_scores = [0.0 if math.isnan(score) else score for score in bert_scores]
        bert_scores = np.mean(bert_scores)
        print(f'{args.dataset} {args.model} BERTScore: {round(bert_scores, 4)}')
    elif args.mode == 'pbert':
        f = open('bert-ptest.txt', 'w')
        # add the BERTScore
        from bert_score import score
        datasets = ['dailydialog', 'empchat', 'personachat']
        models = ['HRED', 'HRAN', 'DSHRED', 'DSHRED_RA', 'ReCoSa', 'ReCoSa_RA', 'WSeq', 'WSeq_RA']
        for dataset in datasets:
            f.write(f'========== {dataset} ==========\n')
            f.flush()
            for model in models:
                if dataset == 'dailydialog' and model != 'WSeq_RA':
                    continue
                f.write(f'========== {model} ==========\n')
                f.flush()
                for i in range(1, 11):
                    context, groundtruth, reply = obtain_test_data(f'./data/{dataset}/{model}-{i}-pred.txt')
        
                    _, _, bert_scores = score(reply, groundtruth, lang='en',
                                              rescale_with_baseline=True)
                    bert_scores = bert_scores.tolist()
                    bert_scores = [0.0 if math.isnan(score) else score for score in bert_scores]
                    bert_scores = np.mean(bert_scores)
                    f.write(f'{dataset} {model}-{i} BERTScore: {round(bert_scores, 4)}\n')
                    f.flush()

