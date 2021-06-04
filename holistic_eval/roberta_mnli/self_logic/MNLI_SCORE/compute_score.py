import json
from scipy import stats
modes = ['seq2seq','word_net','baseline']
results = dict()
for mode in modes:
    sample_file = 'self_logic/MNLI_SCORE/data/{}/samples.json'.format(mode)
    score_file = 'self_logic/MNLI_SCORE/data/{}/mnli.pred'.format(mode)
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
    results[mode] = MNLI_scores
with open('self_logic/labeled/system_score.json' , 'w', encoding='utf8') as f:
    json.dump(results, f)
    # a = stats.spearmanr(MNLI_scores, human_scores)
    # b = stats.pearsonr(MNLI_scores, human_scores)
    # print(a)
    # print(b)
    # x=1
