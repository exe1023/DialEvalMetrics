import os
import re
import json
import numpy as np
from scipy import stats

def compute_score(data_dir):
    files = os.listdir(data_dir)
    scores = []
    for i , file in enumerate(files):
        t_scores = []
        with open(data_dir + '/' + file, encoding='utf8') as f:
            for j, line in enumerate(f):
                if 'YOUR SCORE' in line:
                    t_scores.append(float(re.findall("\d+",line)[0]))
        if t_scores==[]:
            x=1
        scores.append(t_scores)
    if scores == []:
        x=1
    return scores

from_xlx = dict()
from_xlx['baseline'] = '3	3	3	4	5	4	5	3	5	4	4	3	3	5	3	3	4	3	3	4	3	5	3	2	5	5	3	4	5	3	3	4	5	3	4	3	3	3	4	2	3	3	4	5	4	2	4	3	5	4	5	3	3	5	3	5	5	5	5	3	4	2	5	4	5	4	4	3	4	4	3	2	3	3	4	5	3	4	2	3	3	1	1	5	4	5	5	4	2	3	2	4	4	4	5	4	5	1	4	4'.split()
from_xlx['seq2seq'] = '3	1	4	1	3	5	5	4	5	4	5	3	2	2	3	1	5	5	5	5	5	3	2	4	4	5	1	1	4	2	1	3	2	2	5	3	2	3	4	2	5	4	5	5	1	5	5	3	5	1	1	5	4	5	5	3	5	3	1	1	5	2	5	5	2	1	3	3	3	2	5	5	3	3	5	3	5	4	1	5	4	3	3	1	3	5	5	1	2	3	3	4	3	5	5	3	3	2	4	1'.split()
from_xlx['word_net'] = '4	3	4	4	4	3	1	5	5	1	3	3	3	4	3	3	4	2	4	5	5	3	4	1	2	4	4	3	3	3	3	4	5	3	5	3	2	1	2	2	2	2	4	3	2	4	5	5	5	3	2	3	3	4	3	5	5	5	3	3	3	2	5	3	3	2	1	3	3	2	3	2	5	1	3	3	3	4	2	3	3	4	1	3	2	5	4	1	2	4	3	5	4	3	5	3	2	1	3	3'.split()
labels = ['baseline',  'word_net', 'seq2seq']
with open('self_logic/labeled/system_score.json', encoding='utf8') as f:
    system_scores = json.load(f)
for label in labels:
    data_dir = os.path.join('self_logic/labeled',label)
    human_scores = compute_score(data_dir)
    from_xlx[label] = [float(a) for a in from_xlx[label]]
    human_scores.append(from_xlx[label])
    avg_human_scores = np.array(human_scores).mean(0)
    baseline_system_score = system_scores[label]


    spearmanr_score = stats.spearmanr(avg_human_scores, baseline_system_score)
    pearsonr_score = stats.pearsonr(avg_human_scores, baseline_system_score)
    print('average pearsonr score for {} is {:.4f}'.format(label, pearsonr_score[0]))
    print('average spearmanr score for {} is {:.4f}'.format(label, spearmanr_score[0]))


    inter_spearmanr_scores = []
    inter_pearsonr_scores = []
    for i in range(len(human_scores)):
        t_score = human_scores[i]
        other_score = human_scores[:i] + human_scores[i+1:]
        other_avg_score = np.array(other_score).mean(0)
        inter_spearmanr_scores.append(stats.spearmanr(t_score, other_avg_score)[0])
        inter_pearsonr_scores.append(stats.pearsonr(t_score, other_avg_score)[0])
    print(
        'Inter pearsonr (mean) for {} is {:.4f}'.format(label, sum(inter_pearsonr_scores) / len(inter_pearsonr_scores)))
    print('Inter pearsonr (max) for {} is {:.4f}'.format(label, max(inter_pearsonr_scores)))
    print('Inter spearman (mean) for {} is {:.4f}'.format(label,
                                                          sum(inter_spearmanr_scores) / len(inter_spearmanr_scores)))
    print('Inter spearman (max) for {} is {:.4f}'.format(label, max(inter_spearmanr_scores)))

    human_scores = np.array(human_scores)
    print('Variance of {} is {}'.format(label, np.var(human_scores,0).mean()))

    dist_to_avg = []
    for i in range(human_scores.shape[0]):
        dist_to_avg.append(np.linalg.norm(human_scores[i] - avg_human_scores))
    x=1



# baseline_dir = 'labeled/baseline'
# seq2seq_dir = 'labeled/seq2seq'
# wn_dir = 'labeled/wn'
# hum_dirs = [baseline_dir, seq2seq_dir, wn_dir]
#
#
#
# baseline_scores = compute_score(baseline_dir)
# seq2seq_scores = compute_score(seq2seq_dir)
# wn_scores = compute_score(wn_dir)
# x=1
# with open('baseline2_100_samples/system_score_baseline.json', encoding ='utf8') as f:
#     baseline_system_score = json.load(f)
# with open('sample_to_MNLI_data_100samples/system_score_seq2seq.json', encoding ='utf8') as f:
#     seq2seq_system_score = json.load(f)
# with open('sample_to_word_para_data_100samples/system_score_wn.json', encoding ='utf8') as f:
#     wn_system_score = json.load(f)
# x=1
# avg_baseline_score = np.array(baseline_scores).mean(0)
# avg_seq2seq_scores = np.array(seq2seq_scores).mean(0)
# avg_wn_scores = np.array(wn_scores).mean(0)
# a = stats.spearmanr(avg_baseline_score, baseline_system_score)
# b = stats.pearsonr(avg_baseline_score, baseline_system_score)
