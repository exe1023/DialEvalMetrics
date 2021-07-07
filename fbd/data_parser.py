from pathlib import Path
import json

def gen_fbd_data(data, target_dir):

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    if 'models' not in data.keys():
        with open(f'{target_dir}/data.json', 'w') as f:
            for context, ref, hyp in zip(data['contexts'], data['references'], data['responses']):
                f.write(
                    json.dumps(
                        {'src': ' '.join(context), 
                        'refs': [ref],
                        'hyps': [hyp],
                        'human_scores': [[0]]
                        }
                    ) + '\n'
                )
    else:
        context2data = {}
        for context, ref, hyp, model, score in zip(data['contexts'], data['references'], data['responses'], data['models'], data['scores']):
            context = ' '.join(context)
            
            if isinstance(score, dict):
                score = score['Overall']

            if context not in context2data:
                context2data[context] = {
                    'ref': ref,
                    'hyp': {model: hyp},
                    'score': {model: score}
                }
            else:
                context2data[context]['hyp'][model] = hyp
                context2data[context]['score'][model] = score

        models = list(set(data['models']))

        with open(f'{target_dir}/data.json', 'w') as f:
            for context, data in context2data.items():
                f.write(
                    json.dumps(
                        {'src': context,
                        'refs': [ref],
                        'hyps': [data['hyp'][x] if x in data['hyp'] else 'NO HYP' for x in models],
                        'human_scores': [[data['score'][x]] if x in data['score'] else [0] for x in models]
                        }
                    ) + '\n'
                )
            


    
def read_deb_result(data_dir):
    with open(f'{data_dir}/result.json') as f:
        score = json.load(f)

    return [x[0] for x in score]