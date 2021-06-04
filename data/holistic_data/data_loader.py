import csv
import json
from pathlib import Path
import numpy as np

def load_holistic_data(base_dir):
    base_dir = Path(base_dir)
    
    contexts, responses, references, scores = [], [], [], []
    with (base_dir / 'context_data_release.csv').open() as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            context = row[1] 
            response = row[2]
            human_scores = [int(x) for x in row[3:]]
            score = np.mean(human_scores)

            contexts.append([context])
            responses.append(response)
            references.append('NO REF')
            scores.append(score)


    return {
        'contexts': contexts,
        'responses': responses,
        'references': references,
        'scores': scores
    }



if __name__ == '__main__':
    data = load_holistic_data('.')
    print(data['contexts'][:5])
    print(data['responses'][:5])
    print(data['scores'][:5])
    with open('holistic_context.json', 'w') as f:
        json.dump(data, f)