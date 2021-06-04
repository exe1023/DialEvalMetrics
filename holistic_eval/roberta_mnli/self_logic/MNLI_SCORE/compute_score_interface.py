import json
from scipy import stats

def compute_score_intf(sample_file, score_file):
    results = dict()

    with open(sample_file, encoding='utf8') as f:
        samples = json.load(f)
    with open(score_file, encoding='utf8') as f:
        scores = f.readlines()
        scores = [a.strip().split('\t')[-1] for a in scores]

    score_id = 0
    MNLI_scores = []
    # human_scores = []
    for sample in samples:
        sample['scores'] = []
        for i in range(len(sample['pairs'])):
            sample['scores'].append(float(scores[score_id]))
            score_id += 1
        MNLI_scores.append((1-sample['scores'][-1])*5)
        # MNLI_scores.append((1 - sum(sample['scores'])/len(sample['scores'])) * 5)
        # MNLI_scores.append((1 - max(sample['scores']) / len(sample['scores'])) * 5)
        # human_scores.append(sample['humen_score'])
    return MNLI_scores



    # a = stats.spearmanr(MNLI_scores, human_scores)
    # b = stats.pearsonr(MNLI_scores, human_scores)
    # print(a)
    # print(b)
    # x=1
