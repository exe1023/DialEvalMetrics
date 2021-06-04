import json
from pathlib import Path

def gen_amfm_data(data, target_dir):
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
def read_amfm_result(data_dir):
    results = {}
    with open(f'{data_dir}/am_scores.json') as f:
        result = json.load(f)
        results['AM'] = [x[0] for x in result['scores']]
    with open(f'{data_dir}/fm_scores.json') as f:
        result = json.load(f)
        results['FM'] = [x[0] for x in result['scores']]
    
    results['AMFM'] = [0.5 * x + 0.5 * y for x, y in zip(results['AM'], results['FM'])]
    return results