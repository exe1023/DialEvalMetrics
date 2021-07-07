import argparse
import json
from pathlib import Path

from data.grade_data.data_loader import load_grade_data
from maude.data_parser import read_maude_result
from grade.data_parser import read_grade_result
from ruber.data_parser import read_ruber_result, read_bert_ruber_result, write_ruber_result
from holistic_eval.data_parser import read_hostilic_result
from predictive_engagement.data_parser import read_engagement_result
from am_fm.data_parser import read_amfm_result
from FlowScore.data_parser import read_flowscore_result
from usl_dialogue_metric.data_parser import read_usl_result
from questeval.data_parser import read_questeval_result
from deb.data_parser import read_deb_result
from dynaeval.data_parser import read_dynaeval_result
from dialogrpt.data_parser import read_dialogrpt_result

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--eval_data', type=str, default=None)
    args = parser.parse_args()
    return args

def basic_write_result(target, metric, scores):
    Path(target).mkdir(parents=True, exist_ok=True)
    with open(f'{target}/results.json', 'w') as fout:
        json.dump({metric: scores}, fout)

def direct_write_result(target, metric, scores):
    Path(target).mkdir(parents=True, exist_ok=True)
    with open(f'{target}/results.json', 'w') as fout:
        json.dump(scores, fout)

def main(eval_data, metric):
    format_type = 0 # default
    write_result = basic_write_result
    if metric == 'maude':
        data_path = 'maude/eval_data'
        read_result = read_maude_result
    elif metric == 'grade':
        data_path = 'grade/evaluation/infer_result'
        read_result = read_grade_result
        format_type = 1
    elif metric == 'ruber':
        data_path = 'ruber_and_bert_ruber/RUBER/data'
        read_result = read_ruber_result
        write_result = write_ruber_result
    elif metric == 'bert_ruber':
        data_path = 'PONE/PONE/data'
        read_result = read_bert_ruber_result
        write_result = write_ruber_result
    elif metric == 'holistic':
        data_path = 'holistic_eval/eval_data'
        read_result = read_hostilic_result
        write_result = direct_write_result
    elif metric == 'predictive_engagement':
        data_path = 'predictive_engagement/data'
        read_result = read_engagement_result
    elif metric == 'amfm':
        data_path = 'am_fm/examples/dstc6/test_data'
        read_result = read_amfm_result
        write_result = direct_write_result
    elif metric == 'flowscore':
        data_path = 'FlowScore/results'
        read_result = read_flowscore_result
    elif metric == 'usl':
        data_path = 'usl_dialogue_metric/usl_score/datasets'
        read_result = read_usl_result
    elif metric == 'questeval':
        data_path = 'questeval/outputs'
        read_result = read_questeval_result
    elif metric == 'deb':
        data_path = 'deb/dataset'
        read_result = read_deb_result
    elif metric == 'dynaeval':
        data_path = 'dynaeval/data'
        read_result = read_dynaeval_result
    elif metric == 'dialogrpt':
        data_path = 'dialogrpt/test_data'
        read_result = read_dialogrpt_result
    else:
        raise Exception
    '''
    Adding a new metric
    elif metric == 'metric':
        data_path = 'PATH/TO/OUTPUT/DIR'
        read_result = read_metric_result # the customized function
    '''
    
    if eval_data == 'convai2_grade':
        model_names = ['bert_ranker', 'dialogGPT', 'transformer_generator', 'transformer_ranker']
        for model in model_names:
            scores = read_result(f'{data_path}/convai2_grade_{model}')
            target_dir = f'outputs/{metric}/grade_data/convai2/{model}'
            write_result(target_dir, metric, scores)

    elif eval_data == 'dailydialog_grade':
        model_names = ['transformer_generator', 'transformer_ranker']
        for model in model_names:
            scores = read_result(f'{data_path}/dailydialog_grade_{model}')
            target_dir = f'outputs/{metric}/grade_data/dailydialog/{model}'
            write_result(target_dir, metric, scores)

    elif eval_data == 'empatheticdialogues_grade':
        model_names = ['transformer_generator', 'transformer_ranker']
        for model in model_names:
            scores = read_result(f'{data_path}/empatheticdialogues_grade_{model}')
            target_dir = f'outputs/{metric}/grade_data/empatheticdialogues/{model}'
            write_result(target_dir, metric, scores)

    elif eval_data == 'personachat_usr':
        if format_type == 0:
            scores = read_result(f'{data_path}/personachat_usr')
        elif format_type ==1:
            scores = read_result(f'{data_path}/eval_personachat_usr/model')
        target_dir = f'outputs/{metric}/usr_data/personachat'
        write_result(target_dir, metric, scores)

    elif eval_data == 'topicalchat_usr':
        if format_type == 0:
            scores = read_result(f'{data_path}/topicalchat_usr')
        elif format_type ==1:
            scores = read_result(f'{data_path}/eval_topicalchat_usr/model')
        target_dir = f'outputs/{metric}/usr_data/topicalchat'
        write_result(target_dir, metric, scores)

    else:
        if format_type == 0:
            scores = read_result(f'{data_path}/{eval_data}')
        elif format_type ==1:
            scores = read_result(f'{data_path}/eval_{eval_data}/model')
        
        target_dir = f'outputs/{metric}/{eval_data}_data'
        write_result(target_dir, metric, scores)
       
if __name__ == '__main__':
    args = parse_args()
    if args.metric is not None:
        metrics = [args.metric]
    else:
        metrics = ['maude', 'grade', 'ruber', 'bert_ruber', 'holistic', 'predictive_engagement', 'amfm', 'flowscore',
        'usl', 'questeval', 'dialogrpt']
    
    if args.eval_data is not None:
        eval_data = [args.eval_data]
    else:
        eval_data = ['convai2_grade', 'dailydialog_grade', 'empatheticdialogues_grade',
                    'personachat_usr', 'topicalchat_usr', 'dstc6', 'fed', 'dstc9', 'engage', 'fed_dialog', 'holistic']

    for data in eval_data:
        for metric in metrics:
            print(f'Reading {data} from {metric}')

            main(data, metric)
            try:
                main(data, metric)
            except:
                print(f"Cant find {data} {metric}")