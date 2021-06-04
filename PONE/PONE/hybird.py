#!/usr/bin/python
# Author: GMFTBY, sfs, hyx
# Time: 2019.7.10


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from rouge import Rouge

import argparse
import os
import pickle
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm
import random
import scipy
import csv
# from metrics.cider import cider
from scipy.stats.mstats import gmean
from scipy.stats import pearsonr, spearmanr
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
# from bert_score import score

from reference_score import *
from unreference_score import *
from utils import *

import json

# need the nlg-eval, download may cost lots of time, try to set the flag to false
# from nlgeval import compute_individual_metrics
# from nlgeval import compute_metrics


def load_word_embedding(path, dimension=300):
    # load chinese or english word embedding
    unk = np.random.rand(dimension)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    dic = {}
    for line in tqdm(lines):
        dic[line.split()[0]] = np.array([float(f) for f in line.strip().split()[1:]], dtype=np.float)
    dic['<unk>'] = unk
    return dic


def collection_result(contextp, groundp, predp, cembedpath, gembedpath, rembedpath):
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
    cembed, gembed, rembed = read_file(cembedpath), read_file(gembedpath), read_file(rembedpath)
    # cembed = cembed[:100]
    # gembed = gembed[:100]
    # rembed = rembed[:100]
    return context, groundtruth, reply, cembed, gembed, rembed


# ========== fuck nlg-eval fuck ========== #
# ========== Our own embedding-based metric ========== #
def cal_vector_extrema(x, y, dic):
    # x and y are the list of the words
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                vectors.append(dic[w])
            else:
                vectors.append(dic['<unk>'])
        if not vectors:
            vectors.append(dic['<unk>'])
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    vec_x = np.max(x, axis=0)
    vec_y = np.max(y, axis=0)
    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    zero_list = np.zeros(len(vec_x))
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)
    res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos


def cal_embedding_average(x, y, dic):
    # x and y are the list of the words
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                vectors.append(dic[w])
            else:
                vectors.append(dic['<unk>'])
        if not vectors:
            vectors.append(dic['<unk>'])
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    
    vec_x = np.array([0 for _ in range(len(x[0]))])  # 存放句向量
    for x_v in x:
        x_v = np.array(x_v)
        vec_x = np.add(x_v, vec_x)
    vec_x = vec_x / math.sqrt(sum(np.square(vec_x)))
    
    vec_y = np.array([0 for _ in range(len(y[0]))])  # 存放句向量
    #print(len(vec_y))
    for y_v in y:
        y_v = np.array(y_v)
        vec_y = np.add(y_v, vec_y)
    vec_y = vec_y / math.sqrt(sum(np.square(vec_y)))
    
    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    zero_list = np.array([0 for _ in range(len(vec_x))])
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)
    res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos


def cal_greedy_matching(x, y, dic):
    # x and y are the list of words
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                vectors.append(dic[w])
            else:
                vectors.append(dic['<unk>'])
        if not vectors:
            vectors.append(dic['<unk>'])
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    
    len_x = len(x)
    len_y = len(y)
    
    #计算greedy(x,y)
    cosine = []  # 存放一个句子的一个词与另一个句子的所有词的余弦相似度
    sum_x = 0  # 存放最后得到的结果

    for x_v in x:
        for y_v in y:
            assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
            zero_list = np.zeros(len(x_v))

            if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                if x_v.all() == y_v.all():
                    cos = float(1)
                else:
                    cos = float(0)
            else:
                # method 1
                res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

            cosine.append(cos)
        if cosine:
            sum_x += max(cosine)
            cosine = []

    sum_x = sum_x / len_x
    #计算greedy(y,x)
    cosine = []

    sum_y = 0  # 存放最后得到的结果

    for y_v in y:

        for x_v in x:
            assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
            zero_list = np.zeros(len(y_v))

            if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                if (x_v == y_v).all():
                    cos = float(1)
                else:
                    cos = float(0)
            else:
                # method 1
                res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

            cosine.append(cos)

        if cosine:
            sum_y += max(cosine)
            cosine = []

    sum_y = sum_y / len_y
    score = (sum_x + sum_y) / 2
    return score

# ========== End of our own embedding-based metric ========== #


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
    try:
        if not refer:
            refer = ['...']
        return sentence_bleu(refer, candidate, weights=weight, smoothing_function=smoothie)
    except:
        ipdb.set_trace()

def cal_ROUGE(refer, candidate):
    if not candidate:
        candidate = 'unk'
    rouge = Rouge()
    try:
        if not refer:
            refer = ['...', '...']
        scores = rouge.get_scores(' '.join(candidate), ' '.join(refer))
    except:
        ipdb.set_trace()
    return scores[0]['rouge-2']['f']
    

def show(scores1, scores2, scores3, model_scores, mode, aspect):
    print(f'========== Method {mode}:{aspect} result ==========')
    def subshow(scores, m):
        if aspect == 'fluency':
            human = [i[0] for i in scores]
        elif aspect == 'coherence':
            human = [i[1] for i in scores]
        elif aspect == 'safety':
            human = [i[2] for i in scores]
        elif aspect == 'overall':
            human = [i[3] for i in scores]
        else:
            raise Exceptio(f'[!] wrong aspect: {aspect}, should be (fluency, coherence, safety, overall)')

        if 'Human' in mode:
            if aspect == 'fluency':
                model_scores = [i[0] for i in model_scores]
            elif aspect == 'coherence':
                model_scores = [i[1] for i in model_scores]
            elif aspect == 'safety':
                model_scores = [i[2] for i in model_scores]
            elif aspect == 'overall':
                model_scores = [i[3] for i in model_scores]
            else:
                raise Exceptio(f'[!] wrong aspect: {aspect}, should be (fluency, coherence, safety, overall)')
        p, pp = pearsonr(human, m)
        p, pp = round(p, 5), round(pp, 5)
        s, ss = spearmanr(human, m)
        s, ss = round(s, 5), round(ss, 5)
        return p, pp, s, ss
    
    p, pp, s, ss = [], [], [], []
    p1, pp1, s1, ss1 = subshow(scores1, model_scores)
    p.append(p1)
    pp.append(pp1)
    s.append(s1)
    ss.append(ss1)
    
    p2, pp2, s2, ss2 = subshow(scores2, model_scores)
    p.append(p2)
    pp.append(pp2)
    s.append(s2)
    ss.append(ss2)
    
    p3, pp3, s3, ss3 = subshow(scores3, model_scores)
    p.append(p3)
    pp.append(pp3)
    s.append(s3)
    ss.append(ss3)
    
    p, pp, s, ss = round(np.max(p), 5), round(np.min(pp), 5), round(np.max(s), 5), round(np.min(ss), 5)
    
    print('Pearson(p-value):', f'{p}({pp})')
    print('Spearman(p-value):', f'{s}({ss})')
    print(f'========== Method {mode}:{aspect} result ==========')
    
    return p, pp, s, ss


def show_human(scores, model_scores, mode, aspect):
    if aspect == 'fluency':
        human = [i[0] for i in scores]
    elif aspect == 'coherence':
        human = [i[1] for i in scores]
    elif aspect == 'safety':
        human = [i[2] for i in scores]
    elif aspect == 'overall':
        human = [i[3] for i in scores]
    else:
        raise Exceptio(f'[!] wrong aspect: {aspect}, should be (fluency, coherence, safety, overall)')

    if 'Human' in mode:
        if aspect == 'fluency':
            model_scores = [i[0] for i in model_scores]
        elif aspect == 'coherence':
            model_scores = [i[1] for i in model_scores]
        elif aspect == 'safety':
            model_scores = [i[2] for i in model_scores]
        elif aspect == 'overall':
            model_scores = [i[3] for i in model_scores]
        else:
            raise Exceptio(f'[!] wrong aspect: {aspect}, should be (fluency, coherence, safety, overall)')

    p, pp = pearsonr(human, model_scores)
    p, pp = round(p, 5), round(pp, 5)
    s, ss = spearmanr(human, model_scores)
    s, ss = round(s, 5), round(ss, 5)
    return p, pp, s, ss
    
    
    
def read_human_score(paths):
    def read_file(path):
        with open(path) as f:
            score = []
            for line in f.readlines():
                score.append(float(line.strip()))
        return score
    score = []
    for path in paths:
        try:
            score.append(read_file(path))
        except:
            continue
    return score

def read_human_score_csv(paths):
    def read_csv_file(path):
        with open(path, encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter=',')
            score = []
            for line in f_csv:
                if len(line) == 1:
                    # only test the overall score
                    score.append((0.5, 0.5, 0.5, float(line[0])))
                else:
                    if line[0].startswith('\ufeff'):
                        line[0] = line[0][1:]
                    score.append((float(line[0]), float(line[1]), float(line[2]), float(line[3])))
        return score
    score = []
    for path in paths:
        try:
            score.append(read_csv_file(path))
        except:
            print(f'[!] read file {path} error')
            raise Exception()
    return score


class BERT_RUBER:
    
    def __init__(self, dataset, model_name, 
                 bert_size=768, epoch_threshold=1, test=False):
        self.mode = test
        self.refer = BERT_RUBER_refer(test=test)
        self.unrefer = BERT_RUBER_unrefer(bert_size)
        
        #load_best_model(dataset, model_name, self.unrefer, threshold=epoch_threshold)
        get_best_model(args.pretrain_data, model_name, self.unrefer, threshold=epoch_threshold)
        
        if torch.cuda.is_available():
            self.unrefer.cuda()
            self.unrefer.eval()
            
    def normalize(self, scores):
        smin = min(scores)
        smax = max(scores)
        diff = smax - smin
        ret = [(s - smin) / diff for s in scores]
        return ret
    
    def refer_score(self, groundtruth, reply):
        refer_score = self.refer.cos_similarity(groundtruth, reply)
        return refer_score
    
    def unrefer_score(self, query, reply):
        # query / reply: [B, H], B may be the 1 for evaluation
        query = torch.from_numpy(query)
        reply = torch.from_numpy(reply)
        if torch.cuda.is_available():
            q, r = query.cuda(), reply.cuda()
        
        unrefer_score = self.unrefer(q, r)    # [B]
        unrefer_score = unrefer_score.cpu().detach().numpy()    # [B]   
        return unrefer_score
    
    def scores(self, cembed, gembed, rembed, method='Min'):
        # get unreference score
        unrefer = self.unrefer_score(cembed, rembed)
        
        # get reference score
        refer = []
        for g, r in zip(gembed, rembed):
            refer_s = cos_similarity(g, r)
            refer.append(refer_s)
            
        refer = self.normalize(refer)
        unrefer = self.normalize(unrefer)
        ruber = self.hybird_score(refer, unrefer, method=method)
        
        return refer, unrefer, ruber
    
    def hybird_score(self, refer, unrefer, method='Min'):
        # make sure refer and unrefer has been normed
        if method == 'Min':
            return [min(a,b) for a,b in zip(refer, unrefer)]
        elif method == 'Max':
            return [max(a,b) for a,b in zip(refer, unrefer)]
        else:
            raise Exception("Can not find the right method")
            
    def test(self, c, g, r, method='Min'):
        # test function for evaluation
        # c, g, r are the sentence without segmentation
        if not self.mode:
            raise Exception('[!] Error: set the test parameter to True and make sure bert-as-service is running')
        else:
            refer_score = self.refer.cos_similarity(g, r)
            # unrefer score, [1, 3072]
            cembed = self.refer.encode_sentence(c).reshape(1, -1)
            rembed = self.refer.encode_sentence(r).reshape(1, -1)
            unrefer_score = self.unrefer_score(cembed, rembed)[0]
            ruber = self.hybird_score([refer_score], [unrefer_score], method=method)[0]
        return refer_score, unrefer_score, ruber
    
    
# aggregation the scores of the fluent_best, coherence_best, safe_best
def aggregate_scores(fluent_model, coherence_model, safe_model, 
                     context, groundtruth, reply, cembed, gembed, rembed,
                     f_alpha=0.25, c_alpha=0.5, s_alpha=0.25):
    fluent_m = BERT_RUBER(args.dataset, fluent_model, bert_size=args.bert_size,
                          epoch_threshold=args.epoch_threshold, test=args.test)
    coherence_m = BERT_RUBER(args.dataset, coherence_model, bert_size=args.bert_size,
                          epoch_threshold=args.epoch_threshold, test=args.test)
    safe_m = BERT_RUBER(args.dataset, safe_model, bert_size=args.bert_size,
                          epoch_threshold=args.epoch_threshold, test=args.test)
    
    bleu1, bleu2, bleu3, bleu3, bleu4, rouge2, refer, f_unrefer, f_ruber = model_get_score(context, groundtruth, reply, cembed, gembed, rembed, model_name=fluent_model)
    _, _, _, _, _, _, _, c_unrefer, c_ruber = model_get_score(context, groundtruth, reply, cembed, gembed, rembed, model_name=coherence_model)
    _, _, _, _, _, _, _, s_unrefer, s_ruber = model_get_score(context, groundtruth, reply, cembed, gembed, rembed, model_name=safe_model)
    
    # aggregate the unrefer scores with the hyperparameters
    unrefer = f_alpha * f_unrefer + c_alpha * c_unrefer + s_alpha * s_unrefer
    
    return bleu1, bleu2, bleu3, bleu4, rouge2, refer, unrefer, c_ruber
    
    
def get_ruber_score(context, groundtruth, reply, model_name='original_model'):  
    model = BERT_RUBER(args.dataset, model_name, 
                bert_size=args.bert_size,
                epoch_threshold=args.epoch_threshold,
                test=args.test)
    refers, unrefer, ruber = model.scores(cembed, gembed, rembed, method='Min')
    return refers, unrefer, ruber
            
def model_get_score(context, groundtruth, reply, 
                    cembed, gembed, rembed, dic,
                    model_name='original_model'):
    '''
    [!] Get the scores of one model
    '''
    # init the model
    model = BERT_RUBER(args.dataset, model_name, 
                       bert_size=args.bert_size,
                       epoch_threshold=args.epoch_threshold,
                       test=args.test)
    
    # init the metric collectors
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    rouge2_scores = []
    vector_extrema_scores, embedding_average_scores, greedy_matching_scores = [], [], []
    
    # BERT RUBER
    refers, unrefer, ruber = model.scores(cembed, gembed, rembed, method='Min')
    cider_scores, meteor_scores = [], []
    
    # BLEU and rouge and embedding-based
    for c, g, r in tqdm(zip(context, groundtruth, reply)):
        # cider and meteor for english
        # nlgscores = compute_individual_metrics([g], r, no_overlap=False,
        #                                        no_skipthoughts=True,
        #                                        no_glove=True)
        # cider_scores.append(nlgscores['CIDEr'])
        # meteor_scores.append(nlgscores['METEOR'])
        
        # bleu, rough and embedding-based
        refer, condidate = g.split(), r.split()
        bleu1_scores.append(cal_BLEU(refer, condidate, ngram=1))
        bleu2_scores.append(cal_BLEU(refer, condidate, ngram=2))
        bleu3_scores.append(cal_BLEU(refer, condidate, ngram=3))
        bleu4_scores.append(cal_BLEU(refer, condidate, ngram=4))
        rouge2_scores.append(cal_ROUGE(refer, condidate))
        try:
            vector_extrema_scores.append(cal_vector_extrema(refer, condidate, dic))
            embedding_average_scores.append(cal_embedding_average(refer, condidate, dic))
            greedy_matching_scores.append(cal_greedy_matching(refer, condidate, dic))
        except Exception as e:
            ipdb.set_trace()

        
    # BERTScore, too slow to run it
    # if args.dataset in ['xiaohuangji', 'tencent']: language = 'zh'
    # else: language = 'en'
    # print(f'Language: {language}')
    # ipdb.set_trace()
    # _, _, bert_scores = score(reply, groundtruth, lang=language)
    # bert_scores = bert_scores.tolist()
    # nbert_scores = []
    # for i in bert_scores:
    #     if math.isnan(i):
    #         nbert_scores.append(0.5)
    #     else:
    #         nbert_scores.append(i)
    # bert_scores = nbert_scores
    
    print(f'[!] finish computing the score of {model_name}')
    return bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, rouge2_scores, vector_extrema_scores, embedding_average_scores, greedy_matching_scores, refers, unrefer, ruber
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EPN-RUBER hybird test script')
    parser.add_argument('--weight', dest='weight', action='store_true')
    parser.add_argument('--no-weight', dest='weight', action='store_false')
    parser.add_argument('--seed', type=int, default=123, 
                        help='seed for random init')
    parser.add_argument('--model_name', type=str, default='big-no-weight',
                        help='model name of different versions')
    parser.add_argument('--pretrain_data', type=str, default='dailydialog')
    parser.add_argument('--epoch_threshold', type=int, default=30,
                        help='the threshold of epoch for loading best performance model')
    parser.add_argument('--bert_size', type=int, default=768,
                        help='the bert embedding size (3072(-4, -3, -2, -1) for REDUCE_MEAN / REDUCE_MAX or 1536 for REDUCE_MAX_MEAN)')
    parser.add_argument('--dataset', type=str, default='xiaohuangji',
                        help='the dataset we used')
    parser.add_argument('--human_annotator', nargs='+', default=['./data/xiaohuangji/person1-xiaohuangji-rest.txt',
                                                                 './data/xiaohuangji/person2-xiaohuangji-rest.txt',
                                                                 './data/xiaohuangji/person3-xiaohuangji-rest.txt'],
                        help='the human annotation file we need')
    parser.add_argument('--output', type=str, default='./data/xiaohuangji/result.txt', 
                        help='avg performance into the output file')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')

    args = parser.parse_args()

    # set the random seed for the model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # ========== calculate the scores ==========
    # load the test dataset
    # seq2seq-attn: '300 for tencent and xiaohuangji'; copynet 100 for tencent and xiaohuangji
    if args.dataset in ['tencent', 'xiaohuangji', 'dailydialog_wseq', 'dailydialog_hred', 'dailydialog_recosa']:
        sample_num = 300
    else:
        sample_num = 100    # NOTE: chinese 300; english 100
    '''
    context, groundtruth, reply, cembed, gembed, rembed = collection_result(f'./data/{args.dataset}/sample-{sample_num}.txt', 
                                                                            f'./data/{args.dataset}/sample-{sample_num}-tgt.txt',
                                                                            f'./data/{args.dataset}/pred.txt',
                                                                            f'./data/{args.dataset}/sample-src.embed',
                                                                            f'./data/{args.dataset}/sample-tgt.embed', 
                                                                            f'./data/{args.dataset}/pred.embed')
    '''
    context, groundtruth, reply, cembed, gembed, rembed = collection_result(f'./data/{args.dataset}/context.txt', 
                                                                            f'./data/{args.dataset}/reference.txt',
                                                                            f'./data/{args.dataset}/hypothesis.txt',
                                                                            f'./data/{args.dataset}/context.embed',
                                                                            f'./data/{args.dataset}/reference.embed', 
                                                                            f'./data/{args.dataset}/hypothesis.embed')


    refers, unrefer, ruber = get_ruber_score(context, groundtruth, reply, model_name=args.model_name)
    refers = [float(v) for v in refers]
    unrefer = [float(v) for v in unrefer]
    ruber = [float(v) for v in ruber]
    with open(f'data/{args.dataset}/refer_score_{args.pretrain_data}.json', 'w') as f:
        json.dump(refers, f)

    with open(f'data/{args.dataset}/unrefer_score_{args.pretrain_data}.json', 'w') as f:
        json.dump(unrefer, f)

    with open(f'data/{args.dataset}/ruber_score_{args.pretrain_data}.json', 'w') as f:
        json.dump(ruber, f)
    exit(0)

    cembed = cembed[:sample_num]
    gembed = gembed[:sample_num]
    rembed = rembed[:sample_num]
    
    print(f'[!] read file over')
    
    if args.dataset in ['xiaohuangji', 'tencent', 'weibo']:
        lang = True
    elif args.dataset in ['twitter', 'dailydialog', 'cornell', 'dailydialog_hred',
                          'dailydialog_recosa', 'dailydialog_wseq']:
        lang = False
    else:
        raise Exception(f'[!] ERROR: Unknown dataset {args.dataset}')
        
    # load direction of the english and chinese for embedding-based metrics
    if lang:
        if os.path.exists('./data/chinese.pkl'):
            with open('./data/chinese.pkl', 'rb') as f:
                dic = pickle.load(f)
        else:
            dic = load_word_embedding('/home/lt/data/File/wordembedding/chinese/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5')
            with open('./data/chinese.pkl', 'wb') as f:
                pickle.dump(dic, f)
        print('[!] end of loading chinese embeddings')
    else:
        if os.path.exists('./data/english.pkl'):
            with open('./data/english.pkl', 'rb') as f:
                dic = pickle.load(f)
        else:
            dic = load_word_embedding('/home/lt/data/File/wordembedding/glove/glove.6B.300d.txt')
            with open('./data/english.pkl', 'wb') as f:
                pickle.dump(dic, f)
        print('[!] end of loading english embeddings')
    
    # calculate the scores by the single model
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, rouge2_scores, vector_extrema_scores, embedding_average_scores, greedy_matching_scores, refers, unrefer, ruber = model_get_score(context, groundtruth, reply, 
                        cembed, gembed, rembed, dic,
                        model_name=args.model_name)
    
    # calculate the scores by three models aggregation
    # bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, rouge2_scores, refers, unrefers, ruber = aggregate_scores(args.fluent_model, args.coherence_model, args.safe_model,
    #                      context, groundtruth, reply, 
    #                      cembed, gembed, rembed)
    print(f'[!] finish compute the scores of the model(s)')
    
    # load human scores
    print(f'[!] Load human annotator from files, {args.human_annotator}')
    # h: [file, 300]
    h = read_human_score_csv(args.human_annotator)
    
    # human scores
    h1, h2, h3 = h
    
    hfluency, hcoherence, hsafety, hoverall = [], [], [], []
    hfluency.append(show_human(h1, h2, 'Human 1v2', 'fluency'))
    hcoherence.append(show_human(h1, h2, 'Human 1v2', 'coherence'))
    hsafety.append(show_human(h1, h2, 'Human 1v2', 'safety'))
    hoverall.append(show_human(h1, h2, 'Human 1v2', 'overall'))
        
    hfluency.append(show_human(h1, h3, 'Human 1v3', 'fluency'))
    hcoherence.append(show_human(h1, h3, 'Human 1v3', 'coherence'))
    hsafety.append(show_human(h1, h3, 'Human 1v3', 'safety'))
    hoverall.append(show_human(h1, h3, 'Human 1v3', 'overall'))
        
    hfluency.append(show_human(h2, h3, 'Human 2v3', 'fluency'))
    hcoherence.append(show_human(h2, h3, 'Human 2v3', 'coherence'))
    hsafety.append(show_human(h2, h3, 'Human 2v3', 'safety'))
    hoverall.append(show_human(h2, h3, 'Human 2v3', 'overall'))
    
    # ========== human max and avg ===========
    print('========== Method Human avg:fluency result==========')
    print(f'Pearson(p-value): {round(np.mean([i[0] for i in hfluency]), 5)}({round(np.mean([i[1] for i in hfluency]), 5)})')
    print(f'Spearman(p-value): {round(np.mean([i[2] for i in hfluency]), 5)}({round(np.mean([i[3] for i in hfluency]), 5)})')
    print('========== Method Human avg:fluency result==========')
    print('========== Method Human avg:coherence result==========')
    print(f'Pearson(p-value): {round(np.mean([i[0] for i in hcoherence]), 5)}({round(np.mean([i[1] for i in hcoherence]), 5)})')
    print(f'Spearman(p-value): {round(np.mean([i[2] for i in hcoherence]), 5)}({round(np.mean([i[3] for i in hcoherence]), 5)})')
    print('========== Method Human avg:coherence result==========')
    print('========== Method Human avg:safety result==========')
    print(f'Pearson(p-value): {round(np.mean([i[0] for i in hsafety]), 5)}({round(np.mean([i[1] for i in hsafety]), 5)})')
    print(f'Spearman(p-value): {round(np.mean([i[2] for i in hsafety]), 5)}({round(np.mean([i[3] for i in hsafety]), 5)})')
    print('========== Method Human avg:safety result==========')
    print('========== Method Human avg:overall result==========')
    print(f'Pearson(p-value): {round(np.mean([i[0] for i in hoverall]), 5)}({round(np.mean([i[1] for i in hoverall]), 5)})')
    print(f'Spearman(p-value): {round(np.mean([i[2] for i in hoverall]), 5)}({round(np.mean([i[3] for i in hoverall]), 5)})')
    print('========== Method Human avg:overall result==========')
    print('========== ========== ==========')
    print('========== ========== ==========')
    print('========== Method Human max:fluency result==========')
    print(f'Pearson(p-value): {round(np.max([i[0] for i in hfluency]), 5)}({round(np.min([i[1] for i in hfluency]), 5)})')
    print(f'Spearman(p-value): {round(np.max([i[2] for i in hfluency]), 5)}({round(np.min([i[3] for i in hfluency]), 5)})')
    print('========== Method Human max:fluency result==========')
    print('========== Method Human max:coherence result==========')
    print(f'Pearson(p-value): {round(np.max([i[0] for i in hcoherence]), 5)}({round(np.min([i[1] for i in hcoherence]), 5)})')
    print(f'Spearman(p-value): {round(np.max([i[2] for i in hcoherence]), 5)}({round(np.min([i[3] for i in hcoherence]), 5)})')
    print('========== Method Human max:coherence result==========')
    print('========== Method Human max:safety result==========')
    print(f'Pearson(p-value): {round(np.max([i[0] for i in hsafety]), 5)}({round(np.min([i[1] for i in hsafety]), 5)})')
    print(f'Spearman(p-value): {round(np.max([i[2] for i in hsafety]), 5)}({round(np.min([i[3] for i in hsafety]), 5)})')
    print('========== Method Human max:safety result==========')
    print('========== Method Human max:overall result==========')
    print(f'Pearson(p-value): {round(np.max([i[0] for i in hoverall]), 5)}({round(np.min([i[1] for i in hoverall]), 5)}')
    print(f'Spearman(p-value): {round(np.max([i[2] for i in hoverall]), 5)}({round(np.min([i[3] for i in hoverall]), 5)})')
    print('========== Method Human max:overall result==========')
    
    # ========== Word-overlap-based ========== # 
    show(h1, h2, h3, bleu1_scores, "BLEU-1", 'fluency')
    show(h1, h2, h3, bleu1_scores, "BLEU-1", 'coherence')
    show(h1, h2, h3, bleu1_scores, "BLEU-1", 'safety')
    show(h1, h2, h3, bleu1_scores, "BLEU-1", 'overall')
    
    show(h1, h2, h3, bleu2_scores, "BLEU-2", 'fluency')
    show(h1, h2, h3, bleu2_scores, "BLEU-2", 'coherence')
    show(h1, h2, h3, bleu2_scores, "BLEU-2", 'safety')
    show(h1, h2, h3, bleu2_scores, "BLEU-2", 'overall')
    
    show(h1, h2, h3, bleu3_scores, "BLEU-3", 'fluency')
    show(h1, h2, h3, bleu3_scores, "BLEU-3", 'coherence')
    show(h1, h2, h3, bleu3_scores, "BLEU-3", 'safety')
    show(h1, h2, h3, bleu3_scores, "BLEU-3", 'overall')
    
    show(h1, h2, h3, bleu4_scores, "BLEU-4", 'fluency')
    show(h1, h2, h3, bleu4_scores, "BLEU-4", 'coherence')
    show(h1, h2, h3, bleu4_scores, "BLEU-4", 'safety')
    show(h1, h2, h3, bleu4_scores, "BLEU-4", 'overall')
    
    show(h1, h2, h3, rouge2_scores, "ROUGE", 'fluency')
    show(h1, h2, h3, rouge2_scores, "ROUGE", 'coherence')
    show(h1, h2, h3, rouge2_scores, "ROUGE", 'safety')
    show(h1, h2, h3, rouge2_scores, "ROUGE", 'overall')
    
    # show(h1, h2, h3, cider_scores, "CIDEr", 'fluency')
    # show(h1, h2, h3, cider_scores, "CIDEr", 'coherence')
    # show(h1, h2, h3, cider_scores, "CIDEr", 'safety')
    # show(h1, h2, h3, cider_scores, "CIDEr", 'overall')
    
    # show(h1, h2, h3, meteor_scores, "METEOR", 'fluency')
    # show(h1, h2, h3, meteor_scores, "METEOR", 'coherence')
    # show(h1, h2, h3, meteor_scores, "METEOR", 'safety')
    # show(h1, h2, h3, meteor_scores, "METEOR", 'overall')
    
    # ========== embedding-based (vector-extrema, average-embedding, greedy matching) ========== #
    show(h1, h2, h3, vector_extrema_scores, "Vector-Extrema", 'fluency')
    show(h1, h2, h3, vector_extrema_scores, "Vector-Extrema", 'coherence')
    show(h1, h2, h3, vector_extrema_scores, "Vector-Extrema", 'safety')
    show(h1, h2, h3, vector_extrema_scores, "Vector-Extrema", 'overall')
    
    show(h1, h2, h3, embedding_average_scores, "Embedding-Average", 'fluency')
    show(h1, h2, h3, embedding_average_scores, "Embedding-Average", 'coherence')
    show(h1, h2, h3, embedding_average_scores, "Embedding-Average", 'safety')
    show(h1, h2, h3, embedding_average_scores, "Embedding-Average", 'overall')
    
    show(h1, h2, h3, greedy_matching_scores, "Greedy-Matching", 'fluency')
    show(h1, h2, h3, greedy_matching_scores, "Greedy-Matching", 'coherence')
    show(h1, h2, h3, greedy_matching_scores, "Greedy-Matching", 'safety')
    show(h1, h2, h3, greedy_matching_scores, "Greedy-Matching", 'overall')
    
    # show(h1, h2, h3, bertscore, "BERTScore", 'fluency')
    # show(h1, h2, h3, bertscore, 'BERTScore', 'coherence')
    # show(h1, h2, h3, bertscore, 'BERTScore', 'safety')
    # show(h1, h2, h3, bertscore, 'BERTScore', 'overall')
    
    # ========== LEARNING-BASED ========== #
    su_p_f, su_pp_f, su_s_f, su_ss_f = show(h1, h2, h3, unrefer, "BERT s_U", "fluency")
    su_p_c, su_pp_c, su_s_c, su_ss_c = show(h1, h2, h3, unrefer, "BERT s_U", 'coherence')
    su_p_s, su_pp_s, su_s_s, su_ss_s = show(h1, h2, h3, unrefer, "BERT s_U", 'safety')
    su_p_o, su_pp_o, su_s_o, su_ss_o = show(h1, h2, h3, unrefer, "BERT s_U", 'overall')
    
    sr_p_f, sr_pp_f, sr_s_f, sr_ss_f = show(h1, h2, h3, refers, "BERT s_R", "fluency")
    sr_p_c, sr_pp_c, sr_s_c, sr_ss_c = show(h1, h2, h3, refers, "BERT s_R", 'coherence')
    sr_p_s, sr_pp_s, sr_s_s, sr_ss_s = show(h1, h2, h3, refers, "BERT s_R", 'safety')
    sr_p_o, sr_pp_o, sr_s_o, sr_ss_o = show(h1, h2, h3, refers, "BERT s_R", 'overall')
    
    u_p_f, u_pp_f, u_s_f, u_ss_f = show(h1, h2, h3, ruber, "BERT RUBER", 'fluency')
    u_p_c, u_pp_c, u_s_c, u_ss_c = show(h1, h2, h3, ruber, "BERT RUBER", 'coherence')
    u_p_s, u_pp_s, u_s_s, u_ss_s = show(h1, h2, h3, ruber, "BERT RUBER", 'safety')
    u_p_o, u_pp_o, u_s_o, u_ss_o = show(h1, h2, h3, ruber, "BERT RUBER", 'overall')
    
    # rest into file
    print(f'========== write the result into file {args.output} ==========')
    with open(args.output, 'a') as f:
        f.write(f'su_p_f: {su_p_f}({su_pp_f}), su_s_f: {su_s_f}({su_ss_f})' + '\n')
        f.write(f'su_p_c: {su_p_c}({su_pp_c}), su_s_c: {su_s_c}({su_ss_c})' + '\n')
        f.write(f'su_p_s: {su_p_s}({su_pp_s}), su_s_s: {su_s_s}({su_ss_s})' + '\n')
        f.write(f'su_p_o: {su_p_o}({su_pp_o}), su_s_o: {su_s_o}({su_ss_o})' + '\n')
        
        f.write(f'sr_p_f: {sr_p_f}({sr_pp_f}), sr_s_f: {sr_s_f}({sr_ss_f})' + '\n')
        f.write(f'sr_p_c: {sr_p_c}({sr_pp_c}), sr_s_c: {sr_s_c}({sr_ss_c})' + '\n')
        f.write(f'sr_p_s: {sr_p_s}({sr_pp_s}), sr_s_s: {sr_s_s}({sr_ss_s})' + '\n')
        f.write(f'sr_p_o: {sr_p_o}({sr_pp_o}), sr_s_o: {sr_s_o}({sr_ss_o})' + '\n')
        
        f.write(f'u_p_f: {u_p_f}({u_pp_f}), u_s_f: {u_s_f}({u_ss_f})' + '\n')
        f.write(f'u_p_c: {u_p_c}({u_pp_c}), u_s_c: {u_s_c}({u_ss_c})' + '\n')
        f.write(f'u_p_s: {u_p_s}({u_pp_s}), u_s_s: {u_s_s}({u_ss_s})' + '\n')
        f.write(f'u_p_o: {u_p_o}({u_pp_o}), u_s_o: {u_s_o}({u_ss_o})' + '\n')
