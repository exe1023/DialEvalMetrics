import csv

def gen_maude_data(raw_data, output_path):
    fieldnames = ['', 'Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','Unnamed: 0.1.1.1','dialog_id','context_id',
    'context','true_response','context_hash','bert_context','bert_true_response','split']
    write_data = []
    assert len(raw_data['contexts']) == len(raw_data['responses'])
    for idx, (context, response) in enumerate(zip(raw_data['contexts'], raw_data['responses'])):

        write_data.append({
            'dialog_id': idx,
            'context_id': idx,
            'context': '\n'.join(context),
            'true_response': response,
            'bert_context': '[CLS] ' + '[SEP]'.join(context) + ' [SEP]',
            'bert_true_response': f'[CLS] {response} [SEP]',
            'split': 'test'
        })

    with open(output_path, mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in write_data:
            writer.writerow(row)

def read_maude_result(data_path):
    scores = []
    with open(data_path + '.csv') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            scores.append(float(row['na_all_score_20488119']))
    return scores


if __name__ == '__main__':
    data = gen_maude_data({
        'contexts': [['Hi how are you doing ||| I am good. how are you?']],
        'responses': ['Great!']
    }, 'test.csv')
    read_maude_result('eval_data/convai2_grade.csv')