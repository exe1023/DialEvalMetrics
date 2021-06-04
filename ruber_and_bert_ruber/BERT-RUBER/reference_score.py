#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.7.10


#from bert_serving.client import BertClient
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class BERT_RUBER_refer():
    
    def __init__(self):
        #self.bc = BertClient()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    def encode_sentence(self, sent):
        sent = [' '.join(i.split()[-200:]) for i in sent]
        return self.bc.encode(sent)    # [batch, 768]
    
    def encode_query(self, query):
        sentences = query.split('__eou__')
        se = self.bc.encode(sentences)
        return np.sum(se, axis=0)    # [768]

    def encode(self, text_batch):
        encoding = self.tokenizer([text_batch], return_tensors='pt', padding=True, truncation=True) 
        embed = self.model(encoding['input_ids'], attention_mask=encoding['attention_mask'])
        return embed['pooler_output'].squeeze()

    def cos_similarity(self, groundtruth, generated):
        if generated and groundtruth:
            #gr = self.encode_sentence(groundtruth)
            #ge = self.encode_sentence(generated)
            gr = self.encode(groundtruth)
            ge = self.encode(generated)
            sim = self.sim(gr, ge)
            
            #gr, ge = gr.detach().numpy(), ge.detach().numpy()
            #sim2 = np.dot(gr, ge) / (np.linalg.norm(gr) * np.linalg.norm(ge))

        else:
            sim = 0.0
        return sim.item()
        

if __name__ == "__main__":
    refer = BERT_RUBER_refer()
    sim = refer.cos_similarity('大大大', '你 是 谁 啊')
    print(sim)
