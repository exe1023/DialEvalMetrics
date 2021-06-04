import numpy as np
import csv

#def load_holistic_data(base_dir, data_type='context'):

def gen_hostilic_data(raw_data, output_path):
    fieldnames = ['id', 'context', 'response']
    write_data = []
    for idx, (context, response) in enumerate(zip(raw_data['contexts'], raw_data['responses'])):
        write_data.append({
            'id': idx,
            'context': '\n'.join(context),
            'response': response,
        })

    with open(output_path, mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        #writer.writeheader()
        for row in write_data:
            writer.writerow(row)

def read_hostilic_result(data_path):
    context_scores = []
    with open(data_path + '_context_out.csv') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row['3'] == '':
                context_scores.append(0)
            else:
                context_scores.append(float(row['3']))

    fluency_scores = []
    with open(data_path + '_fluency_out.csv') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row['3'] == '':
                fluency_scores.append(0)
            else:
                fluency_scores.append(float(row['3']))
    
    #with open(data_path + '_diversity_out.csv') as f:
    #    reader = csv.DictReader(f)
    #    for i, row in enumerate(reader):
    #        scores[i].append(float(row['3']))

    logic_scores = []
    with open(data_path + '_logic_out.csv') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row['3'] == '':
                logic_scores.append(0)
            else:
                logic_scores.append(float(row['3']))
    
    mean_scores = np.mean([context_scores, fluency_scores, logic_scores], axis=0).tolist()

    return {
        'holistic_context': context_scores,
        'holistic_fluency': fluency_scores,
        'holistic_logic': logic_scores,
        'holistic': mean_scores
    }
    #return [score[0] for score in scores]


if __name__ == '__main__':
    data = gen_maude_data({
        'contexts': [['Hi how are you doing ||| I am good. how are you?']],
        'responses': ['Great!']
    }, 'test.csv')
    read_maude_result('eval_data/convai2_grade.csv')