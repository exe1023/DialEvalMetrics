import os
import argparse
from pathlib import Path
import json
from collections import defaultdict
from nlgeval import NLGEval
from bert_score import BERTScorer
from adem_eval.eval import eval_adam
from bleurt import score
import tensorflow as tf

tf.compat.v1.flags.DEFINE_string('data','','') # magic to solve bleurt error

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()
    return args

def nlgeval_metrics(refs, hyps):
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=[])
    metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L']
    results = defaultdict(list)
    for ref, hyp in zip(refs, hyps):
        metrics_dict = nlgeval.compute_individual_metrics([ref], hyp)
        for metric in metrics:
            results[metric].append(metrics_dict[metric])
    return results

def bert_score(refs, hyps):
    scorer = BERTScorer(lang='en', rescale_with_baseline=True)
    P, R, F1 = scorer.score(hyps, refs)
    return F1.tolist()

def adam(contexts, refs, hyps):
    base_dir = os.getcwd()
    scores = eval_adam(base_dir, contexts, refs, hyps)
    return scores.tolist()

def bleurt(contexts, refs, hyps):
    checkpoint = "bleurt/bleurt/test_checkpoint"
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(refs, hyps)
    return scores

def baseline_metrics(contexts, refs, hyps):
    results = nlgeval_metrics(refs, hyps)
    results['bert_score'] = bert_score(refs, hyps)
    results['adam'] = adam(contexts, refs, hyps)
    results['bleurt'] = bleurt(contexts, refs, hyps)
    return results

def main(data):
    if data == 'convai2_grade':
        model_names = ['bert_ranker', 'dialogGPT', 'transformer_generator', 'transformer_ranker']
        for model in model_names:
            with open(f'baseline_data/convai2_grade_{model}.json') as f:
                data = json.load(f)
            results = baseline_metrics(data['contexts'], data['references'], data['responses'])
            target_dir = f'outputs/baseline/grade_data/convai2/{model}'
            with open(f'{target_dir}/results.json', 'w') as fout:
                json.dump(results, fout)
    elif data == 'dailydialog_grade':
        model_names = ['transformer_generator', 'transformer_ranker']
        for model in model_names:
            with open(f'baseline_data/dailydialog_grade_{model}.json') as f:
                data = json.load(f)
            results = baseline_metrics(data['contexts'], data['references'], data['responses'])
            target_dir = f'outputs/baseline/grade_data/dailydialog/{model}'
            with open(f'{target_dir}/results.json', 'w') as fout:
                json.dump(results, fout) 
    elif data == 'empatheticdialogues_grade':
        model_names = ['transformer_generator', 'transformer_ranker']
        for model in model_names:
            with open(f'baseline_data/empatheticdialogues_grade_{model}.json') as f:
                data = json.load(f)
            results = baseline_metrics(data['contexts'], data['references'], data['responses'])
            target_dir = f'outputs/baseline/grade_data/empatheticdialogues/{model}'
            with open(f'{target_dir}/results.json', 'w') as fout:
                json.dump(results, fout) 

    elif data == 'personachat_usr':
        with open('baseline_data/personachat_usr.json') as f:
            data = json.load(f)
            results = baseline_metrics(data['contexts'], data['references'], data['responses'])
            target_dir = 'outputs/baseline/usr_data/personachat'
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            with open(f'{target_dir}/results.json', 'w') as fout:
                json.dump(results, fout)
    elif data == 'topicalchat_usr':
        with open('baseline_data/topicalchat_usr.json') as f:
            data = json.load(f)
            results = baseline_metrics(data['contexts'], data['references'], data['responses'])
            target_dir = 'outputs/baseline/usr_data/topicalchat'
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            with open(f'{target_dir}/results.json', 'w') as fout:
                json.dump(results, fout) 
    elif data == 'dstc6':
        with open('baseline_data/dstc6.json') as f:
            data = json.load(f)
            results = baseline_metrics(data['contexts'], data['references'], data['responses'])
            target_dir = 'outputs/baseline/dstc6_data/'
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            with open(f'{target_dir}/results.json', 'w') as fout:
                json.dump(results, fout) 

if __name__ == '__main__':
    args = parse_args()
    if args.data is not None:
        data_list = [args.data]
    else:
        data_list = ['convai2_grade', 'dailydialog_grade', 'empatheticdialogues_grade',
                   'personachat_usr', 'topicalchat_usr', 'dstc6']

    for data in data_list:
        print(f'Evaluating {data}')
        main(data)
