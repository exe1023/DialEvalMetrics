from elasticsearch import Elasticsearch
import random
from elasticsearch import helpers
from tqdm import tqdm

'''
Write the dataset into the elasticsearch
'''

class ESUtils:

    def __init__(self, index_name, create_index=False):
        self.es = Elasticsearch()
        self.index = index_name
        if create_index:
            mapping = {
                'properties': {
                    'context': {
                        'type': 'text',
                        'analyzer': 'ik_max_word',
                        'search_analyzer': 'ik_max_word'
                    }
                }
            }
            if self.es.indices.exists(index=self.index):
                print(f'[!] delete the index of the elasticsearch')
                self.es.indices.delete(index=self.index)
            rest = self.es.indices.create(index=self.index)
            print(rest)
            rest = self.es.indices.put_mapping(body=mapping, index=self.index)

    def insert_pairs(self, pairs):
        count = self.es.count(index=self.index)['count']
        actions = []
        for i, res in enumerate(tqdm(pairs)):
            actions.append({
                '_index': self.index,
                '_id': i + count,
                'context': res[1],
                'nidx': res[0],
            })
        helpers.bulk(self.es, actions) 

class ESChat:

    def __init__(self, dataset):
        self.es = Elasticsearch()
        self.index = dataset 
        with open(f'data/{dataset}/tgt-train.txt') as f:
            self.data = f.readlines()
            self.data = [i.strip() for i in self.data]

    def search(self, query, samples=10):
        '''
        query is the string, which contains the utterances of the conversation context.
        1. topic msg
        2. key word msg
        cantenate with the space operator
        '''
        query_text = self.data[query]
        query_text = ' '.join(query_text.split()[-50:])
        dsl = {
            'query': {
                'match': {
                    'context': query_text,
                }
            }
        }
        begin_samples, rest = samples, []
        while len(rest) == 0:
            hits = self.es.search(
                    index=self.index, 
                    body=dsl, 
                    size=begin_samples)
            hits = hits['hits']['hits']
            for h in hits:
                idx = h['_source']['nidx']
                if idx == query:
                    continue
                rest.append(idx)
            begin_samples += 1 
        return random.choice(rest)

if __name__ == "__main__":
    # write the file
    import sys
    import ipdb

    dataset = sys.argv[1]
    print(f'[!] write the {dataset} into the elasticsearch')
    esagent = ESUtils(dataset, create_index=True)

    with open(f'data/{dataset}/tgt-train.txt') as f:
        data = f.readlines()
        data = [(idx, i.strip()) for idx, i in enumerate(data)]

    esagent.insert_pairs(data)
    print(f'[!] database size: {esagent.es.count(index=esagent.index)}')
