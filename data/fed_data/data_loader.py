import json
from pathlib import Path
import numpy as np

def load_fed_data(base_dir):
    base_dir = Path(base_dir)
    
    with (base_dir / 'fed_data.json').open() as f:
        data = json.load(f)

    contexts, references, responses, scores = [], [], [], []
    for idx, sample in enumerate(data):

        context = []
        context_raw = sample['context'].split('\n')
        for text_raw in context_raw:
            text = ':'.join(text_raw.split(':')[1:])
            #text = text_raw.split(': ')[1]
            context.append(text.strip())
        
        try:
            response_raw = sample['response']
        except:
            continue
        
        response = ':'.join(response_raw.split(':')[1:])
        #response = response_raw.split(': ')[1]
        response = response.strip()
        
        score = {}
        annotations = sample['annotations']
        for aspect in annotations.keys():
            aspect_score = [x for x in annotations[aspect] if type(x) == int]
            if len(aspect_score) == 0:
                score[aspect] = 0
            else:
                score[aspect] = np.mean(aspect_score)
    
        contexts.append(context)
        references.append('NO REF')
        responses.append(response)
        scores.append(score)

    return {
        'contexts': contexts,
        'references': references,
        'responses': responses,
        'scores': scores
    }


def load_fed_dialog_data(base_dir):
    base_dir = Path(base_dir)
    
    with (base_dir / 'fed_data.json').open() as f:
        data = json.load(f)

    contexts, references, responses, scores = [], [], [], []
    for idx, sample in enumerate(data):

        if 'response' in sample:
            continue

        context = []
        context_raw = sample['context'].split('\n')
        for text_raw in context_raw:
            text = ':'.join(text_raw.split(':')[1:])
            context.append(text.strip())
        
        context = context[:-1]
        response = context[-1]
        
        score = {}
        annotations = sample['annotations']
        for aspect in annotations.keys():
            aspect_score = [x for x in annotations[aspect] if type(x) == int]
            if len(aspect_score) == 0:
                score[aspect] = 0
            else:
                score[aspect] = np.mean(aspect_score)
    
        contexts.append(context)
        references.append('NO REF')
        responses.append(response)
        scores.append(score)

    return {
        'contexts': contexts,
        'references': references,
        'responses': responses,
        'scores': scores
    }



if __name__ == '__main__':
    data = load_fed_data('.')
    with open('fed_human_score.json', 'w') as f:
        json.dump(data, f)
    print(data['contexts'][:5])
    print(data['responses'][:5])
    print(data['scores'][:5])

    data = load_fed_dialog_data('.')
    with open('fed_dialog_human_score.json', 'w') as f:
        json.dump(data, f)
    print(data['contexts'][:5])
    print(data['responses'][:5])
    print(data['scores'][:5])