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
    with open(f'{data_dir}/refer_score.json') as f:
        results['ruber_refer'] = json.load(f)
    with open(f'{data_dir}/unrefer_score.json') as f:
        results['ruber_unrefer'] = json.load(f)
    with open(f'{data_dir}/ruber_score.json') as f:
        results['ruber'] = json.load(f)
    return results['ruber']