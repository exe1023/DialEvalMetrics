import json
def read_questeval_result(data_dir):
    with open(f'{data_dir}.json') as f:
        data = json.load(f)
        
    return [x['fscore'] for x in data]