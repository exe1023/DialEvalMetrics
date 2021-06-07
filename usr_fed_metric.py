import argparse
import json
import requests
from collections import defaultdict
from pathlib import Path
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=str, default='convai2_grade')
    args = parser.parse_args()
    return args

def run(data_path, usr_target_path, fed_target_path, use_fact=False):
    # Modify server path here
    #usr_server = 'localhost:10235'
    #fed_server = 'localhost:10234'

    with open(data_path) as f:
        data = json.load(f)
        
        datas = []
        for idx, (context, response) in enumerate(zip(data['contexts'], data['responses'])):
            if use_fact and 'facts' in data:
                fact = ' '.join(data['facts'][idx])
            else:
                fact = ''

            sample = {
                'dialogid': idx,
                'dialogue_context': [{'agent': 'agent', 'text': c} for c in context],
                'dialog_fact': fact,
                'response_list': [response],
                'agent_name': 'agent'
            }
            datas.append(sample)

        
        usr_scores = defaultdict(list)
        usr_result = requests.post(usr_server, json=datas).text
        usr_result = json.loads(usr_result)
        for result in usr_result['Results']:
            usr_score = result['response_scores']
            for key, item in usr_score[0].items():
                usr_scores[key].append(item)

        Path(usr_target_path).mkdir(parents=True, exist_ok=True)
        with open(f'{usr_target_path}/results.json', 'w') as fout:
            json.dump(usr_scores, fout)


        fed_scores = defaultdict(list)
        fed_result = requests.post(fed_server, json=datas).text
        fed_result = json.loads(fed_result)

        for result in fed_result['Results']:
            fed_score = result['response_scores']
            score_sum = 0.0
            for key, item in fed_score[0].items():
                fed_scores['fed_' + key].append(item)
                score_sum += item
            fed_scores['fed_overall'].append(score_sum / len(fed_score[0]))

        Path(fed_target_path).mkdir(parents=True, exist_ok=True)
        with open(f'{fed_target_path}/results.json', 'w') as fout:
            json.dump(fed_scores, fout)
        

def main(args):

    usr_result_dir = 'outputs/usr'

    if 'grade' in args.data:
        if 'convai2_grade' in args.data:
            model_names = ['bert_ranker', 'dialogGPT', 'transformer_generator', 'transformer_ranker']
            data_name = 'convai2'
        elif 'dailydialog_grade' in args.data:
            model_names = ['transformer_generator', 'transformer_ranker']
            data_name = 'dailydialog'
        elif 'empatheticdialogues_grade' in args.data:
            model_names = ['transformer_generator', 'transformer_ranker']
            data_name = 'empatheticdialogues'
        else:
            raise Exception

        #for model in model_names:
        words = args.data.replace('.json', '').split('_')
        # check dialogGPT
        if words[-1] == 'dialogGPT':
            model = 'dialogGPT'
        else:
            model = words[-2:]
        run(f'usr_fed_data/{args.data}.json', f'{usr_result_dir}/grade_data/{data_name}/{model}', f'outputs/fed/grade_data/{data_name}/{model}')
    elif 'usr' in args.data:
        if 'personachat' in args.data:
            data_name = 'personachat_usr'
            target = 'personachat'
        elif 'topicalchat' in args.data:
            data_name = 'topicalchat_usr'
            target = 'topicalchat'

        #run(args, f'usr_fed_data/{data_name}.json',  f'{usr_result_dir}/usr_data/{target}', f'outputs/fed/usr_data/{target}')
        usr_result_dir = 'outputs/usr_fact'
        run(f'usr_fed_data/{data_name}.json',  f'{usr_result_dir}/usr_data/{target}', f'outputs/fed/usr_data/{target}', use_fact=True)
    else:
        run(f'usr_fed_data/{args.data}.json', f'{usr_result_dir}/{args.data}_data', f'outputs/fed/{args.data}_data')



if __name__ == '__main__':
    args = parse_args()
    main(args)
