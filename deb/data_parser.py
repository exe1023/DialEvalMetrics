from pathlib import Path
import json

def gen_deb_data(data, target_dir):

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{target_dir}/data.json', 'w') as f:
        idx = 0
        for context, hyp in zip(data['contexts'], data['responses']):
            sample = {
                'id': idx,
                'context': context,
                'positive_responses': [hyp],
                'adversarial_negative_responses': [],
                'random_negative_responses': []
            }
            f.write(json.dumps(sample) + '\n')
            idx += 1
    
def read_deb_result(data_dir):
    with open(f'{data_dir}/result.json') as f:
        score = json.load(f)

    return [x[0] for x in score]