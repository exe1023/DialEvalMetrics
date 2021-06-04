#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.7.10

'''
Unreference model with bert embedding
'''
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class BERT_RUBER_unrefer(nn.Module):
    
    def __init__(self, embedding_size, dropout=0.5):
        super(BERT_RUBER_unrefer, self).__init__()

        self.model = AutoModel.from_pretrained('bert-base-uncased').to('cuda')
        for param in self.model.parameters():
            param.requires_grad = False

        self.M = nn.Parameter(torch.rand(embedding_size, embedding_size))
        self.layer1 = nn.Linear(embedding_size * 2 + 1, 256)
        #self.opt = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 128)
        self.opt = nn.Linear(128, 1)
        
        self.opt_drop = nn.Dropout(p=dropout)
    
    def bert_embed(self, batch):
        q_input_ids, q_tokens, q_attention = batch['q_input_ids'].to('cuda'), batch['q_token_type_ids'].to('cuda'), batch['q_attention_mask'].to('cuda')
        r_input_ids, r_tokens, r_attention = batch['r_input_ids'].to('cuda'), batch['r_token_type_ids'].to('cuda'), batch['r_attention_mask'].to('cuda')
        
        b_size = q_input_ids.shape[0] * q_input_ids.shape[1]
        q_embed = self.model(input_ids=q_input_ids.view(b_size, -1), token_type_ids=q_tokens.view(b_size, -1), attention_mask=q_attention.view(b_size, -1))
        r_embed = self.model(input_ids=r_input_ids.view(b_size, -1), token_type_ids=r_tokens.view(b_size, -1), attention_mask=r_attention.view(b_size, -1))
        q_pooled = q_embed['pooler_output']
        r_pooled = r_embed['pooler_output']
        #q_pooled, _ = torch.max(q_embed['last_hidden_state'], dim=1)
        #r_pooled, _ = torch.max(r_embed['last_hidden_state'], dim=1)
        return q_pooled, r_pooled

        
    def forward(self, batch):
        # query / replty: 768-dim bert embedding
        # [B, H]
        # query: list(str)
        # reply: list(str)
        #TODO: max pooling

        query, reply = self.bert_embed(batch)

        qh = query.unsqueeze(1)    # [B, 1, 768]
        rh = reply.unsqueeze(2)    # [B, 768, 1]
        score = torch.bmm(torch.matmul(qh, self.M), rh).squeeze(2)  # [B, 1]
        qh = qh.squeeze(1)    # [B, H]
        rh = rh.squeeze(2)    # [B, H]

        linear = torch.cat([qh, score, rh], 1)    # [B, 2 * H  + 1]
        linear = torch.tanh(self.layer1(linear))
        linear = torch.tanh(self.layer2(linear))
        linear = torch.tanh(self.layer3(linear))
        linear = self.opt(linear).squeeze(1)

        scores = self.sigmoid(linear)  # [B]
        
        return linear, scores


if __name__ == "__main__":
    unrefer = BERT_RUBER_unrefer(200)