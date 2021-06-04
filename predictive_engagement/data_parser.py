import csv

def gen_engagement_data(raw_data, output_path):
    fieldnames = ['id', 'query', 'reply', 'label']
    write_data = []
    for idx, (context, response) in enumerate(zip(raw_data['contexts'], raw_data['responses'])):
        write_data.append({
            'id': idx,
            'query': ' '.join(context),
            'reply': response,
            'label': -1
        })

    with open(output_path, mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in write_data:
            writer.writerow(row)

def read_engagement_result(data_path):
    scores = []
    with open(data_path + '.csv_score.txt') as f:
        for line in f.readlines():
            score = line.strip().split('===')[-1]
            score = float(score)
            scores.append(score)
    return scores