#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.3.28


'''
This file is the model for automatic evaluation natural language reponse
refer to:
RUBER: An Unsupervised Method for Automatic Evaluation of Open-Domain Dialog Systems
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import random
import numpy as np
# import ipdb
import sys
from utils import *


class RUBER_unrefer(nn.Module):
    
    '''
    refer do not need to train, but unrefer need to train
    '''
    
    def __init__(self, query_inpt_size, reply_inpt_size,
                 embed_size, hidden_size, 
                 dropout=0.5, drop=True, pretrain=False, **args):
        super(RUBER_unrefer, self).__init__()
        self.query_word_embed = nn.Embedding(query_inpt_size, embed_size)
        self.reply_word_embed = nn.Embedding(reply_inpt_size, embed_size)
        self.query_rnn = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self.reply_rnn = nn.GRU(embed_size, hidden_size, bidirectional=True)
        
        self.M = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.media_layer = nn.Linear(hidden_size * 2 + 1, hidden_size)
        self.opt = nn.Linear(hidden_size, 1)
        self.media_drop = nn.Dropout(p=dropout)
        
        self.hidden_size = hidden_size
        self.query_inpt_size = query_inpt_size
        self.reply_inpt_size = reply_inpt_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.drop = drop
        
        # init the weight
        self.init_weight()
        
    def init_weight(self):
        # orthogonal init
        init.orthogonal_(self.query_rnn.weight_hh_l0)
        init.orthogonal_(self.query_rnn.weight_ih_l0)
        self.query_rnn.bias_ih_l0.data.fill_(0.0)
        self.query_rnn.bias_hh_l0.data.fill_(0.0)
        
        init.orthogonal_(self.reply_rnn.weight_hh_l0)
        init.orthogonal_(self.reply_rnn.weight_ih_l0)
        self.reply_rnn.bias_ih_l0.data.fill_(0.0)
        self.reply_rnn.bias_hh_l0.data.fill_(0.0)
    
    def forward(self, query, in_lengths, reply, out_lengths):
        '''
        :param query: [T, B]
        :param in_lengths: [B]
        :param reply: [T', B]
        :param out_lengths: [B]
        
        :return linear: [B]
        '''
        #query = query.permute(1, 0)
        #reply = reply.permute(1, 0)
        batch_size = query.shape[1]
        #import ipdb; ipdb.set_trace()
        
        query = self.query_word_embed(query)    # [T, B, E]
        # in_lengths, inidx = torch.sort(in_lengths, descending=True)
        # p = query.transpose(0, 1)[inidx].transpose(0, 1)    # [T, B, E]
        # query = nn.utils.rnn.pack_padded_sequence(p, in_lengths)# qhidden: [1, B, E]
        qoutput, qh = self.query_rnn(query)
        qh = qh[:1, :, :] + qh[1:, :, :]    # [1, B, H]
        # qh = qh.squeeze(0)    # [B, H]
        # qhidden = torch.ones(batch_size, self.hidden_size).cuda()
        # qhidden[inidx] = qh     # recover the indice sort
        # qhidden = qhidden.unsqueeze(0)    # [1, B, H]
        
        reply = self.reply_word_embed(reply)    # [T, B, E]
        # out_lengths, idx = torch.sort(out_lengths, descending=True)
        # p = reply.transpose(0, 1)[idx].transpose(0, 1)    # [T, B, E]
        # reply = nn.utils.rnn.pack_padded_sequence(p, out_lengths)
        routput, rh = self.reply_rnn(reply)
        rh = rh[:1, :, :] + rh[1:, :, :]    # [1, B, H]
        # rh = rh.squeeze(0)    # [B, H]
        # rhidden = torch.ones(batch_size, self.hidden_size).cuda()
        # rhidden[idx] = rh     # recover the indice sort
        # rhidden = rhidden.unsqueeze(0)    # [1, B, H]
            
        
        # qhidden [B, 1, H] / rhidden: [B, H, 1], M: [H, H]
        qh = qh.transpose(0, 1)
        rh = rh.transpose(0, 1).transpose(1, 2)
        score = torch.bmm(torch.matmul(qh, self.M), rh).squeeze(2)  # [B, 1]
        qh = qh.squeeze(1)    # [B, H]
        rh = rh.squeeze(2)    # [B, H]
        linear = torch.cat([qh, score, rh], 1)    # [B, 2 * H  + 1]
        linear = self.media_drop(torch.tanh(self.media_layer(linear)))    # [B, H]
        linear = torch.sigmoid(self.opt(linear).squeeze(1))    # [B]
        return linear
    
if __name__ == "__main__":
    pass