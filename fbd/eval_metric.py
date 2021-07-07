import re
import os
import argparse
import json
import random
import torch
import bert_score
from fbd_score import *
from prd_score import *
from baseline import cal_bleu, cal_meteor, cal_rouge, cal_greedy_match, cal_embd_average, cal_vec_extr
import math
from scipy.stats import spearmanr, pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='dialogue', help='[dialogue | mt]')
parser.add_argument('--data_path', type=str, help='path to dialogue annotation data')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--src_path', type=str, help='path to MT sources')
parser.add_argument('--ref_path', type=str, help='path to MT references')
parser.add_argument('--hyp_path', type=str, help='path to MT hypotheses')
parser.add_argument('--human_path', type=str, help='path to human annotations')
parser.add_argument('--metric', type=str, help='[bleu | meteor | rouge | greedy | average | extrema | bert_score | fbd | prd]')
parser.add_argument('--sample_num', type=int, default=10, help='sample number of references')
parser.add_argument('--model_type', type=str, default='', help='pretrained model type or path to pretrained model')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--is_chinese', type=int, default=0, help='Is Chinese corpus or not')
args = parser.parse_args()

def read_mt_data(args):
    querys, refs, hyps, human_scores = [], [], [], []

    with open(args.src_path, 'r', encoding='utf-8') as f:
        for line in f:
            querys.append(line.strip())
    with open(args.ref_path, 'r', encoding='utf-8') as f:
        for line in f:
            refs.append([line.strip()])

    files = os.listdir(args.hyp_path)
    system_list = []
    for file_ in files:
        hyps.append([])
        system = re.findall(r'news\w*\.(.*)\.\w{2}\-\w{2}', file_)[0]
        system_list.append(system)
        with open(os.path.join(args.hyp_path, file_), 'r', encoding='utf-8') as f:
            for line in f:
                hyps[-1].append(line.strip())
                
    human_scores = [[0 for _ in range(len(system_list))]]
    with open(args.human_path, 'r', encoding='utf-8') as f:
        for line in f:
            system, score = line.split()
            human_scores[0][system_list.index(system)] = float(score)

    return querys, refs, hyps, human_scores
    

def read_dialogue_data(path):
    querys = []
    refs = []
    hyps = []
    human_scores = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            querys.append(line['src'])
            refs.append(line['refs'])

            for i, hyp in enumerate(line['hyps']):
                if len(hyps) < i + 1:
                    hyps.append([])
                hyps[i].append(hyp)

            for i, scores in enumerate(line['human_scores']):
                if len(human_scores) < i + 1:
                    human_scores.append([])
                for j, score in enumerate(scores):
                    if len(human_scores[i]) < j + 1:
                        human_scores[i].append([])
                    human_scores[i][j].append(score)
    
    return querys, refs, hyps, human_scores

def sample(lists, num):
    for i in range(len(lists)):
        if num < len(lists[i]):
            lists[i] = random.sample(lists[i], num)
    return lists

def average(lists):
    for i in range(len(lists)):
        lists[i] = [sum(lst) / len(lst) for lst in lists[i]]
    return lists

def eval_metric(args):
    if args.task_type == 'dialogue':
        querys, refs, hyps, human_scores = read_dialogue_data(args.data_path)
        average_human_scores = average(human_scores)
        human_scores = []
        for scores in average_human_scores:
            for i, score in enumerate(scores):
                if len(human_scores) < i + 1:
                    human_scores.append([])
                human_scores[i].append(score)
    else:
        querys, refs, hyps, human_scores = read_mt_data(args)

    refs = sample(refs, args.sample_num)

    system_scores = []
    print("#-------------------------------------#")
    print(args.metric, args.model_type)
    print("#-------------------------------------#")
    assert args.metric in ['rouge', 'meteor', 'greedy', 'average', 'extrema', 'bert_score', 'fbd', 'prd', 'bleu']
    if args.metric == 'bert_score':
        for hyp in hyps:
            score = bert_score.score(hyp, refs, model_type=args.model_type, batch_size=args.batch_size)
            score = score[2].mean(dim=0).cpu().item()
            system_scores.append(score)

    elif args.metric == 'bleu':
        for hyp in hyps:
            system_scores.append(cal_bleu(refs, hyp, args.is_chinese))

    elif args.metric == 'meteor':
        for hyp in hyps:
            system_scores.append(cal_meteor(refs, hyp))

    elif args.metric == 'rouge':
        for hyp in hyps:
            system_scores.append(cal_rouge(refs, hyp))

    elif args.metric == 'greedy':
        system_scores = cal_greedy_match(refs, hyps)

    elif args.metric == 'average':
        system_scores = cal_embd_average(refs, hyps)

    elif args.metric == 'extrema':
        system_scores = cal_vec_extr(refs, hyps)

    else:
        source_querys = querys
        source_answer_list = hyps
        target_querys, target_answers = [], []
        for query, answers in zip(querys, refs):
            for answer in answers:
                target_querys.append(query)
                target_answers.append(answer)
        
        tokenizer, model = get_model_configs(args.model_type, args.is_chinese)

        if args.metric == 'fbd':
            mu1, sigma1 = get_statistics(target_querys, target_answers, tokenizer, 
                                         model, args.batch_size, use_cuda=True)
                                         #use_tukey_trans=args.use_tukey_trans)
            for source_answers in source_answer_list:
                mu2, sigma2 = get_statistics(source_querys, source_answers, tokenizer, 
                                             model, args.batch_size, use_cuda=True)
                                             #use_tukey_trans=args.use_tukey_trans)
                score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
                system_scores.append(score)

        elif args.metric == 'prd':
            tgt_feats = get_embeddings(target_querys, target_answers, tokenizer, 
                                       model, args.batch_size, use_cuda=True)
            for source_answers in source_answer_list:
                src_feats = get_embeddings(source_querys, source_answers, tokenizer, 
                                           model, args.batch_size, use_cuda=True)
                precision, recall = compute_prd_from_embedding(src_feats, tgt_feats, enforce_balance=False)
                precision = precision.tolist()
                recall = recall.tolist()
                max_f1_score = max([2*p*r/(p+r + 1e-6) for p,r in zip(precision, recall)])
                system_scores.append(max_f1_score)

        else:
            raise NotImplementedError("We don't support the metric: {}".format(args.metric))

    with open(f'{args.data_dir}/result.json', 'w') as f:
        json.dump(system_scores, f)

    pearson_corrs = []
    spearman_corrs = []
    print(human_scores, system_scores)

    for scores in human_scores:
        pearson_corrs.append(pearsonr(system_scores, scores))
        spearman_corrs.append(spearmanr(system_scores, scores))
        #pearson_corrs.append(abs(pearsonr(system_scores, scores)[0]))
        #spearman_corrs.append(abs(spearmanr(system_scores, scores)[0]))
    print('The pearson correlation between {} and human score is {}'.format(args.metric, pearson_corrs))
    print('The spearman correlation between {} and human score is {}'.format(args.metric, spearman_corrs))

if __name__ == '__main__':
    eval_metric(args)