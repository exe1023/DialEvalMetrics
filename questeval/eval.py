import json
import argparse
from questeval.questeval_metric import QuestEval

ref_data = [
   'personachat_usr',
   'topicalchat_usr',
   'convai2_grade_bert_ranker',
   'convai2_grade_transformer_generator',
   'dailydialog_grade_transformer_ranker',
   'empatheticdialogues_grade_transformer_ranker',
   'convai2_grade_dialogGPT',
   'convai2_grade_transformer_ranker',
   'dailydialog_grade_transformer_generator',
   'empatheticdialogues_grade_transformer_generator',
   'dstc6'
]

unref_data = [
   'fed',
   'dstc9',
   'fed_dialog',
   'engage',
   'holistic'
]
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    with open(f'test_data/{args.data}.json') as f:
        data = json.load(f)
    questeval = QuestEval(isCuda=True)
    
    scores = []
    if args.data in ref_data:
        for ctx, ref, hyp in zip(data['contexts'], data['references'], data['responses']):
            score = questeval.compute_all(hyp, ' '.join(ctx), ref)
            scores.append(score['scores'])
    elif args.data in unref_data:
        for ctx, hyp in zip(data['contexts'], data['responses']):
            score = questeval.compute_all(hyp, ' '.join(ctx))
            scores.append(score['scores'])

    else:
        raise Exception

    with open(f'outputs/{args.data}.json', 'w') as f:
        json.dump(scores, f)