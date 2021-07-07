import pickle
from pathlib import Path
import csv
import json

from sklearn.preprocessing import minmax_scale     
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.tokenizer


def gen_dynaeval_data(data, target_dir):

    Path(target_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{target_dir}/data.csv', 'w') as f:
        writer = csv.writer(f, delimiter='|')
        for context, hyp in zip(data['contexts'], data['responses']):
            context.append(hyp)

            dialog = []
            for x in context:
                x = tokenizer(x.lower())
                x = ' '.join([xx.text for xx in x])
                dialog.append(x)
            writer.writerow(
                    ['0',
                    '0 0 0 0 0 0 0 0 0 0 0',
                    dialog,
                    '0 0 0 0 0 0 0 0 0 0 0',
                    dialog]
            )

def read_dynaeval_result(data_dir):
    metric_name = data_dir.split('/')[-1]
    with open(f'{data_dir}/{metric_name}_eval.score', 'rb') as f:
        score = pickle.load(f)
    score = minmax_scale(score)     
    return score.tolist()