import csv
import json
from pathlib import Path
import numpy as np

def load_engage_data(base_dir):
    base_dir = Path(base_dir)
    
    contexts, responses, references, scores = [], [], [], []
    '''with (base_dir / 'engage_data.csv').open() as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:

            for i in range(1, 11):
                context = row['Input.query'+str(i)]
                response = row['Input.response'+str(i)]
                score = row['Answer.pair'+str(i)]

                contexts.append([context])
                responses.append(response)
                references.append('NO REF')
                scores.append(float(score))
    '''

    with (base_dir / 'engage_all.csv').open() as f:
        reader = csv.DictReader(f, delimiter=',')
        rows = []
        for row in reader:
            rows.append(row)
        rows = sorted(rows, key=lambda x: x['query'])
        
        for i in range(0, len(rows), 3):

            context = rows[i]['query'].strip()
            assert len(set([x['query'] for x in rows[i:i+3]])) == 1

            response = rows[i]['response'].strip()
            
            score = np.mean([float(x['human_score']) for x in rows[i:i+3]])

            if context == '':
                context = 'NA'
            if response == '':
                response = 'NA'

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
    data = load_engage_data('.')
    print(len(data['contexts']))
    print(data['contexts'][:5])
    print(data['responses'][:5])
    print(data['scores'][:5])
    with open('engage_data.json', 'w') as f:
        json.dump(data, f)