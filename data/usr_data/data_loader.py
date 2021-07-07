import json
from pathlib import Path
import numpy as np

def load_usr_data(base_dir, dataset):
    base_dir = Path(base_dir)
    
    file = 'pc_usr_data.json' if dataset == 'personachat' else 'tc_usr_data.json'
    
    with (base_dir / file).open() as f:
        data = json.load(f)

    contexts, facts, responses, references, scores, models = [], [], [], [], [], []
    for sample in data:
        context = sample['context']
        fact = sample['fact']
        if dataset == 'personachat':
            fact = fact.replace('your persona: ', ' ')
        # get ground truth
        for response in sample['responses']:
            if response['model'] == 'Original Ground Truth':
                reference = response['response']
                break
        
        for response in sample['responses']:
            if response['model'] == 'Original Ground Truth':
                continue
            contexts.append(context.strip().split('\n'))
            facts.append(fact.strip().split('\n'))
            references.append(reference)
            responses.append(response['response'].strip())
            models.append(response['model'])
            
            scores.append({
                'Understandable': np.mean(response['Understandable']),
                'Natural': np.mean(response['Natural']),
                'Maintains_Context': np.mean(response['Maintains Context']),
                'Engaging': np.mean(response['Engaging']),
                'Uses_Knowledge': np.mean(response['Uses Knowledge']),
                'Overall': np.mean(response['Overall'])
            })

    return {
        'contexts': contexts,
        'facts': facts,
        'responses': responses,
        'references': references,
        'scores': scores,
        'models': models
    }



if __name__ == '__main__':
    data = load_usr_data('.', 'personachat')
    #with open('pc_human_score.txt', 'w') as f:
    #    for score in data['scores']:
    #        f.write(str(score))
    #        f.write('\n')
    with open('pc_parsed.json', 'w') as f:
        json.dump(data, f)

    data = load_usr_data('.', 'topicalchat')
    with open('tc_parsed.json', 'w') as f:
        json.dump(data['scores'], f)
    #print(data['contexts'][:5])
    print(data['facts'][:5])
    print(data['responses'][:5])
    print(data['scores'][:5])