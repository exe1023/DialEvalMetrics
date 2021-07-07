from pathlib import Path
import json

def gen_usl_data(data, target_dir):

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{target_dir}/context.txt', 'w') as f:
        for context in data['contexts']:
            f.write(' '.join(context))
            f.write('\n')

    with open(f'{target_dir}/hypothesis.txt', 'w') as f:
        for hyp in data['responses']:
            f.write(f'{hyp.strip()}\n')
    
def read_usl_result(data_dir):
    usl_scores = []
    with open(f'{data_dir}/score.json') as f:
        for line in f.readlines():
            result = json.loads(line)
            usl_scores.append(result['USL-HS'])
    return usl_scores