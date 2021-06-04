from flow_score import *
import numpy as n
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval_data', type=str, default=None)
    parser.add_argument('--output', type=str)
    
    args = parser.parse_args()
    return args

def main(args):
    FLOW_SCORE = FlowScore(MODEL_PATH)

    with open(args.eval_data) as f:
        data = json.load(f)
        flow_scores = []
        for context, response in zip(data['contexts'], data['responses']):
            flow_input = context + [response]
            flow_score = FLOW_SCORE.score(flow_input) * -1
            flow_scores.append(flow_score)
        data['flow_scores'] = flow_scores

    with open(args.output, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':

    args = parse_args()
    main(args)