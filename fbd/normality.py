import os
import json
import random
import numpy as np
from scipy.stats import (
    jarque_bera,
    shapiro,
    kstest,
    anderson
)
import argparse

from utils import get_model_configs, get_embeddings

"""
CUDA_VISIBLE_DEVICES=1 python normality.py --data_path ./dataset/daily_annotation.json --model_type bert-base-uncased
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

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

data_path = './dataset'
datasets = ["persona", "daily_hybrid", "convai2", "em_hybrid", "daily", "pc"]
models = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]

results = []
for da in datasets:
    for mm in models:
        file_path = os.path.join(data_path, '{}_annotation.json'.format(da))
        querys, refs, hyps, human_scores = read_dialogue_data(file_path)
        refs = sample(refs, 10)

        target_querys, target_answers = [], []
        for query, answers in zip(querys, refs):
            for answer in answers:
                target_querys.append(query)
                target_answers.append(answer)

        tokenizer, model = get_model_configs(mm)
        tgt_feats = get_embeddings(
            target_querys, target_answers, tokenizer, 
            model, args.batch_size, use_cuda=True, 
            use_tukey_trans=False)

        # Shapiro-Wilk test
        all_scores = []
        for i in range(tgt_feats.shape[1]):
            all_scores.append(shapiro(tgt_feats[:,i].flatten()))
        print(da, mm, np.mean(all_scores), np.std(all_scores))
        results.append([da, mm, np.mean(all_scores), np.std(all_scores)])
print(results)


