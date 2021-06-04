import json
from pathlib import Path
def gen_ruber_data(data, target_dir):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{target_dir}/context.txt', 'w') as f:
        for context in data['contexts']:
            f.write(' '.join(context))
            f.write('\n')
    with open(f'{target_dir}/reference.txt', 'w') as f:
        for reference in data['references']:
            f.write(f'{reference.strip()}\n')
    with open(f'{target_dir}/hypothesis.txt', 'w') as f:
        for hyp in data['responses']:
            f.write(f'{hyp.strip()}\n')

def read_ruber_result(data_dir):
    results = {}
    pretrain_data = ['dailydialog', 'personachat']
    for pretrain in pretrain_data:
        with open(f'{data_dir}/refer_score_{pretrain}.json') as f:
            results[f'ruber_refer_{pretrain}'] = json.load(f)
        with open(f'{data_dir}/unrefer_score_{pretrain}.json') as f:
            results[f'ruber_unrefer_{pretrain}'] = json.load(f)
        with open(f'{data_dir}/ruber_score_{pretrain}.json') as f:
            results[f'ruber_{pretrain}'] = json.load(f)
    return results

def read_bert_ruber_result(data_dir):
    results = {}
    pretrain_data = ['dailydialog', 'personachat', 'dailydialog_sampled']
    for pretrain in pretrain_data:
        with open(f'{data_dir}/refer_score_{pretrain}.json') as f:
            results[f'bert_ruber_refer_{pretrain}'] = json.load(f)
        with open(f'{data_dir}/unrefer_score_{pretrain}.json') as f:
            results[f'bert_ruber_unrefer_{pretrain}'] = json.load(f)
        with open(f'{data_dir}/ruber_score_{pretrain}.json') as f:
            results[f'bert_ruber_{pretrain}'] = json.load(f)
    return results


def write_ruber_result(target, metric, scores):
    Path(target).mkdir(parents=True, exist_ok=True)
    with open(f'{target}/results.json', 'w') as fout:
        json.dump(scores, fout)
