import argparse
import os
import json

from data.grade_data.data_loader import load_grade_data
from data.usr_data.data_loader import load_usr_data
from data.dstc6_data.data_loader import load_dstc6_data
from data.fed_data.data_loader import load_fed_data, load_fed_dialog_data
from data.dstc9_data.data_loader import load_dstc9_data
from data.holistic_data.data_loader import load_holistic_data
from data.engage_data.data_loader import load_engage_data

from maude.data_parser import gen_maude_data
from grade.data_parser import gen_grade_data
from ruber.data_parser import gen_ruber_data
from holistic_eval.data_parser import gen_hostilic_data
from predictive_engagement.data_parser import gen_engagement_data
from am_fm.data_parser import gen_amfm_data

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_data', type=str, default=None)
    parser.add_argument('--target_format', type=str, default=None)
    args = parser.parse_args()
    return args

def gen_baseline_data(data, data_path):
    with open(data_path, 'w') as fout:
        json.dump(data, fout)

def main(source_data, target_format):
    format_type = 0
    max_words = None
    if target_format == 'maude':
        metric = 'maude'
        output_dir = f'{os.getcwd()}/maude/eval_data'
        gen_data = gen_maude_data
        suffix = '.csv'
        max_words = 500
    elif target_format == 'hostilic':
        metric = 'hostilic'
        output_dir = f'{os.getcwd()}/holistic_eval/eval_data'
        gen_data = gen_hostilic_data 
        suffix = '.csv'
        max_words = 500
    elif target_format == 'baseline':
        metric = 'baseline'
        output_dir = f'{os.getcwd()}/baseline_data'
        gen_data = gen_baseline_data
        suffix = '.json'
    elif target_format == 'usr_fed':
        metric = 'usr_fed'
        output_dir = f'{os.getcwd()}/usr_fed_data'
        gen_data = gen_baseline_data
        suffix = '.json'
        max_words = 500
    elif target_format == 'ruber':
        metric = 'ruber'
        output_dir = f'{os.getcwd()}/ruber_and_bert_ruber/RUBER/data'
        gen_data = gen_ruber_data
        suffix = ''
    elif target_format == 'bert_ruber':
        metric = 'bert_ruber'
        #output_dir = f'{os.getcwd()}/ruber_and_bert_ruber/BERT-RUBER/data'
        output_dir = f'{os.getcwd()}/PONE/PONE/data'
        gen_data = gen_ruber_data
        suffix = ''
    elif target_format == 'grade':
        metric = 'grade'
        output_dir = f'{os.getcwd()}/grade'
        gen_data = gen_grade_data
        suffix = '.json'
        format_type = 1
    elif target_format == 'predictive_engagement':
        metric = 'predictive_engagement'
        output_dir = f'{os.getcwd()}/predictive_engagement/data'
        gen_data = gen_engagement_data
        suffix = '.csv'
    elif target_format =='amfm':
        metric = 'amfm'
        output_dir = f'{os.getcwd()}/am_fm/examples/dstc6/test_data'
        gen_data = gen_amfm_data
        suffix = ''
    elif target_format == 'flow':
        metric = 'flow'
        output_dir = f'{os.getcwd()}/FlowScore/eval_data'
        gen_data = gen_baseline_data
        suffix = '.json'
        


    if source_data == 'convai2_grade':
        model_names = ['bert_ranker', 'dialogGPT', 'transformer_generator', 'transformer_ranker']
        for model in model_names:
            data_path = f'{os.getcwd()}/data/grade_data'
            data = load_grade_data(data_path, 'convai2', model)
            output_path = f'{output_dir}/convai2_grade_{model}{suffix}'
            gen_data(data, output_path)

    elif source_data == 'dailydialog_grade':
        model_names = ['transformer_generator', 'transformer_ranker']
        for model in model_names:
            data_path = f'{os.getcwd()}/data/grade_data'
            data = load_grade_data(data_path, 'dailydialog', model)
            output_path = f'{output_dir}/dailydialog_grade_{model}{suffix}'
            gen_data(data, output_path)

    elif source_data == 'empatheticdialogues_grade':
        model_names = ['transformer_generator', 'transformer_ranker']
        for model in model_names:
            data_path = f'{os.getcwd()}/data/grade_data'
            data = load_grade_data(data_path, 'empatheticdialogues', model)
            output_path = f'{output_dir}/empatheticdialogues_grade_{model}{suffix}'
            gen_data(data, output_path)
    
    elif source_data == 'personachat_usr':
        data_path = f'{os.getcwd()}/data/usr_data'
        data = load_usr_data(data_path, 'personachat')
        
        if format_type == 0:
            output_path = f'{output_dir}/personachat_usr{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            output_path = gen_data(data, output_dir, 'personachat_usr')
    
    elif source_data == 'topicalchat_usr':
        data_path = f'{os.getcwd()}/data/usr_data'
        data = load_usr_data(data_path, 'topicalchat')
        if format_type == 0:
            output_path = f'{output_dir}/topicalchat_usr{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            output_path = gen_data(data, output_dir, 'topicalchat_usr')
    
    elif source_data == 'dstc6':
        data_path = f'{os.getcwd()}/data/dstc6_data'
        data = load_dstc6_data(data_path)
        if format_type == 0:
            output_path = f'{output_dir}/dstc6{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            output_path = gen_data(data, output_dir, 'dstc6')
    
    elif source_data == 'fed':
        data_path = f'{os.getcwd()}/data/fed_data'
        data = load_fed_data(data_path)
        if format_type == 0:
            output_path = f'{output_dir}/fed{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            output_path = gen_data(data, output_dir, 'fed')
    
    elif source_data == 'fed_dialog':
        data_path = f'{os.getcwd()}/data/fed_data'
        data = load_fed_dialog_data(data_path)
        if format_type == 0:
            output_path = f'{output_dir}/fed_dialog{suffix}'
            gen_data(data, output_path)
        elif format_type == 1:
            output_path = gen_data(data, output_dir, 'fed_dialog')
              
    elif source_data == 'dstc9':
        data_path = f'{os.getcwd()}/data/dstc9_data'
        data = load_dstc9_data(data_path, max_words)
        if format_type == 0:
            output_path = f'{output_dir}/dstc9{suffix}'        
            gen_data(data, output_path)
        elif format_type == 1:
            output_path = gen_data(data, output_dir, 'dstc9')

    elif source_data == 'holistic':
        data_path = f'{os.getcwd()}/data/holistic_data'
        data = load_holistic_data(data_path)
        if format_type == 0:
            output_path = f'{output_dir}/holistic{suffix}'        
            gen_data(data, output_path)
        elif format_type == 1:
            output_path = gen_data(data, output_dir, 'holistic')
    
    elif source_data == 'engage':
        data_path = f'{os.getcwd()}/data/engage_data'
        data = load_engage_data(data_path)
        if format_type == 0:
            output_path = f'{output_dir}/engage{suffix}'        
            gen_data(data, output_path)
        elif format_type == 1:
            output_path = gen_data(data, output_dir, 'engage')
        

if __name__ == '__main__':
    args = parse_args()
    all_data = ['convai2_grade', 'dailydialog_grade', 'empatheticdialogues_grade', 
                'personachat_usr', 'topicalchat_usr', 'dstc6', 'fed', 'dstc9', 'engage']

    if args.source_data is not None:
        all_data = [args.source_data]

    if args.target_format is not None:
        metrics = [args.target_format]
    else:
        metrics = ['maude', 'hostilic', 'baseline', 'usr_fed', 'ruber', 'bert_ruber', 'grade', 'predictive_engagement', 'amfm', 'flow']

    for data in all_data:
        for target in metrics:
            print(f'Generating {data} to {target}')
            main(data, target)
