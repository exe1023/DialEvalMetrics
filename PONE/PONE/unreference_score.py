#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.7.10

'''
Unreference model with bert embedding
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class BERT_RUBER_unrefer(nn.Module):
    
    def __init__(self, embedding_size, dropout=0.5):
        super(BERT_RUBER_unrefer, self).__init__()
        
        self.M = nn.Parameter(torch.rand(embedding_size, embedding_size))
        self.layer1 = nn.Linear(embedding_size * 2 + 1, 256)
        self.layer1_drop = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(256, 512)
        self.layer2_drop = nn.Dropout(p=dropout)
        self.layer3 = nn.Linear(512, 128)
        self.layer3_drop = nn.Dropout(p=dropout)
        self.opt = nn.Linear(128, 1)
        
    def forward(self, query, reply):
        # query / replty: 76-dim bert embedding
        # [B, H]
        qh = query.unsqueeze(1)    # [B, 1, 3072]
        rh = reply.unsqueeze(2)    # [B, 3072, 1]
        score = torch.bmm(torch.matmul(qh, self.M), rh).squeeze(2)  # [B, 1]
        qh = qh.squeeze(1)    # [B, H]
        rh = rh.squeeze(2)    # [B, H]
        linear = torch.cat([qh, score, rh], 1)    # [B, 2 * H  + 1]
        linear = self.layer1_drop(torch.relu(self.layer1(linear)))
        linear = self.layer2_drop(torch.relu(self.layer2(linear)))
        linear = self.layer3_drop(torch.relu(self.layer3(linear)))
        linear = torch.sigmoid(self.opt(linear)).squeeze(1)   # [B]
        
        return linear


if __name__ == "__main__":
    unrefer = BERT_RUBER_unrefer(200)