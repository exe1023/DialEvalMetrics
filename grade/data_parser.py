import json
from pathlib import Path

def read_grade_result(data_path):
    with open(f'{data_path}/non_reduced_results.json') as f:
        data = json.load(f)
    for k, v in data.items():
        return v

def gen_grade_data(data, base_dir, dataset):
    # generate preprocess data
    preprocess_path = f'{base_dir}/preprocess/dataset/eval_{dataset}'
    Path(preprocess_path).mkdir(parents=True, exist_ok=True)
    raw_data = []
    for context, response in zip(data['contexts'], data['responses']):
        raw_data.append(context + [response])

    with open(f'{preprocess_path}/data.json', 'w') as f:
        json.dump(raw_data, f)
        
    # generate context-response data
    output_path = f'{base_dir}/evaluation/eval_data/eval_{dataset}/model'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ctx_f = open(f'{output_path}/human_ctx.txt', 'w')
    hyp_f = open(f'{output_path}/human_hyp.txt', 'w')

    for context, response in zip(data['contexts'], data['responses']):
        ctx_f.write('|||'.join(context))
        ctx_f.write('\n')
        hyp_f.write(response)
        hyp_f.write('\n')
    ctx_f.close()
    hyp_f.close()
    return
