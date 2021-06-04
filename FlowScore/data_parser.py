import json

def read_flowscore_result(data_path):
    with open(data_path + '.json') as f:
        data = json.load(f)
    return data['flow_scores']