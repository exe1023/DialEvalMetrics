import argparse
import json
from pathlib import Path

from data.grade_data.data_loader import load_grade_data
from maude.data_parser import read_maude_result
from grade.data_parser import read_grade_result
from ruber_and_bert_ruber.data_parser import read_ruber_result, read_bert_ruber_result, write_ruber_result
from holistic_eval.data_parser import read_hostilic_result
from predictive_engagement.data_parser import read_engagement_result
from am_fm.data_parser import read_amfm_result
from FlowScore.data_parser import read_flowscore_result

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
    else:
        raise Exception
    
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
    
    elif eval_data == 'dstc6':
        if format_type == 0:
            scores = read_result(f'{data_path}/dstc6')
        elif format_type ==1:
            scores = read_result(f'{data_path}/eval_dstc6/model')
        
        target_dir = f'outputs/{metric}/dstc6_data'
        write_result(target_dir, metric, scores)
    
    elif eval_data == 'fed':
        if format_type == 0:
            scores = read_result(f'{data_path}/fed')
        elif format_type ==1:
            scores = read_result(f'{data_path}/eval_fed/model')
        
        target_dir = f'outputs/{metric}/fed_data'
        write_result(target_dir, metric, scores)
    
    elif eval_data == 'fed_dialog':
        if format_type == 0:
            scores = read_result(f'{data_path}/fed_dialog')
        elif format_type ==1:
            scores = read_result(f'{data_path}/eval_fed_dialog/model')
        
        target_dir = f'outputs/{metric}/fed_dialog_data'
        write_result(target_dir, metric, scores)

    elif eval_data == 'dstc9':
        if format_type == 0:
            scores = read_result(f'{data_path}/dstc9')
        elif format_type ==1:
            scores = read_result(f'{data_path}/eval_dstc9/model')
        
        target_dir = f'outputs/{metric}/dstc9_data'
        write_result(target_dir, metric, scores)
    
    elif eval_data == 'holistic':
        if format_type == 0:
            scores = read_result(f'{data_path}/holistic')
        elif format_type ==1:
            scores = read_result(f'{data_path}/eval_holistic/model')
        
        target_dir = f'outputs/{metric}/holistic_data'
        write_result(target_dir, metric, scores) 
         
    elif eval_data == 'engage':
        if format_type == 0:
            scores = read_result(f'{data_path}/engage')
        elif format_type ==1:
            scores = read_result(f'{data_path}/eval_engage/model')
        
        target_dir = f'outputs/{metric}/engage_data'
        write_result(target_dir, metric, scores) 
     
    else:
        raise Exception
         
if __name__ == '__main__':
    args = parse_args()
    if args.metric is not None:
        metrics = [args.metric]
    else:
        metrics = ['maude', 'grade', 'ruber', 'bert_ruber', 'holistic', 'predictive_engagement', 'amfm', 'flowscore']
    
    if args.eval_data is not None:
        eval_data = [args.eval_data]
    else:
        eval_data = ['convai2_grade', 'dailydialog_grade', 'empatheticdialogues_grade',
                    'personachat_usr', 'topicalchat_usr', 'dstc6', 'fed', 'dstc9', 'engage', 'fed_dialog']

    for data in eval_data:
        for metric in metrics:
            print(f'Reading {data} from {metric}')

            #main(data, metric)
            try:
                main(data, metric)
            except:
                print(f"Cant find {data} {metric}")