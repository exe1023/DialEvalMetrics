"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""

import torch
import torch.nn as nn
import os
import numpy as np


def get_mlp(input_dim, output_dim, num_layers=2, dropout=0.0, hidden_dim=None):
    network_list = []
    if hidden_dim is None:
        hidden_dim = input_dim
    assert num_layers > 0
    for _ in range(num_layers - 1):
        network_list.append(nn.Linear(input_dim, hidden_dim))
        network_list.append(nn.ReLU())
        network_list.append(nn.BatchNorm1d(num_features=hidden_dim))
        network_list.append(nn.Dropout(dropout))
        input_dim = hidden_dim
    network_list.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(
        *network_list
    )


class BiLSTMEncoder(nn.Module):
    def __init__(self, args, emb, device=None):
        super(BiLSTMEncoder, self).__init__()
        self.bsize = args.batch_size
        self.word_emb_dim = args.word2vec_embedding_dim
        self.enc_lstm_dim = args.ruber_lstm_dim
        self.pool_type = args.ruber_unref_pooling_type
        self.dpout_model = args.ruber_dropout
        self.device = device
        self.emb = emb
        self.max_pad = False

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=args.ruber_dropout)

    def forward(self, sent_tuple):
        # sent_len [max_len, ..., min_len] (batch)
        # sent (seqlen x batch x worddim)
        sent, sent_len = sent_tuple
        sent = self.emb(sent)
        self.device = sent.device
        sent_len = sent_len.cpu().numpy()

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).to(self.device)
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
        sent_output = sent_output.index_select(1, idx_unsort)
        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).to(self.device)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2
        else:
            raise NotImplementedError("pooling type not implemented")

        return emb


class UnreferencedMetric(nn.Module):
    def __init__(self, args, data, log):
        """
            Initialize related variables and construct the neural network graph.
            Args:
                qmax_length, rmax_length: max sequence length for query and reply
                fqembed, frembed: embedding matrix file for query and reply
                gru_num_units: number of units in each GRU cell
                mlp_units: number of units for mlp, a list of length T,
                    indicating the output units for each perceptron layer.
                No need to specify the output layer size 1.
    """
        super(UnreferencedMetric, self).__init__()
        self.log = log
        self.device = torch.device(args.device)
        self.word_dict = data.word_dict
        self.emb = nn.Embedding(len(self.word_dict), args.word2vec_embedding_dim, padding_idx=0)
        if args.ruber_load_emb:
            self.emb.weight = torch.load(args.word2vec_out)
        self.queryGRU = BiLSTMEncoder(args, self.emb, self.device)
        self.replyGRU = BiLSTMEncoder(args, self.emb, self.device)
        self.quadratic_M = nn.Parameter(torch.zeros((args.word2vec_embedding_dim * 2, args.word2vec_embedding_dim * 2)))
        nn.init.xavier_uniform_(self.quadratic_M)

        self.mlp = get_mlp((args.word2vec_embedding_dim * 4 + 1), 1, 2, hidden_dim=args.ruber_mlp_dim)

    def forward(self, query_batch, query_length, reply_batch, reply_length):
        qout = self.queryGRU((query_batch.transpose(1,0), query_length))  # B x dim * 2
        rout = self.replyGRU((reply_batch.transpose(1,0), reply_length))  # B x dim * 2

        M = self.quadratic_M.unsqueeze(0)
        bM = M.expand(qout.size(0), M.size(1), M.size(2))  # B x dim x dim
        qTM = torch.bmm(qout.unsqueeze(1), bM)  # B x 1 x dim
        quadratic = torch.bmm(qTM, rout.unsqueeze(2))  # B x 1 x 1
        quadratic = quadratic.squeeze(1)  # B x 1
        # quadratic = qTM.mul(rout).sum(1).unsqueeze(1) # B x 1
        xin = torch.cat([qout, quadratic, rout], dim=1)
        out = self.mlp(xin)  # B x (dim * 4 + 1)
        # out = B x 1
        return torch.sigmoid(out)

class ReferencedMetric():
    """Referenced Metric
    Measure the similarity between the groundtruth reply and generated reply
    use cosine score.
    Provide three pooling methods for generating sentence vector:
        [max_min | avg | all]
    """
    def __init__(self, args, data, log):
        """
        Args:
            data_dir:
            fword2vec: word2vec text file
            pooling_type: [max_min | avg | all], default max_min
        """
        self.args = args
        self.log = log
        self.embeddings = nn.Embedding(len(data.word_dict), args.word2vec_embedding_dim)
        if not os.path.exists(args.word2vec_out):
            raise AssertionError("no pretrained word vectors found")
        self.embeddings.weight = torch.load(args.word2vec_out)
        log.write_message_logs("Loaded {} pre-trained word embeddings of dim {}".format(self.embeddings.weight.size(0),
                                                                                        self.embeddings.weight.size(1)))
        if self.args.ruber_ref_pooling_type=='max_min':
            self.pooling = self.max_min_pooling
        elif self.args.ruber_ref_pooling_type=='avg':
            self.pooling = self.average_pooling
        else:
            self.pooling = self.all_pooling

    def sentence_vector(self, sentence):
        # sentence = sentence.rstrip().split()
        return self.embeddings(sentence)

    def max_pool(self, sent):
        sent[sent == 0] = -1e9
        emb = torch.max(sent, 1)[0]
        return emb

    def min_pool(self, sent):
        sent[sent == 0] = 1e9
        emb = torch.min(sent, 1)[0]
        return emb

    def max_min_pooling(self, sentence, sent_len):
        svector = self.sentence_vector(sentence)  # B x w x dim

        maxp = self.max_pool(svector)           # B x dim
        minp = self.min_pool(svector)           # B x dim
        return torch.cat([maxp, minp], dim=1)    # B x dim*2

    def average_pooling(self, sentence, sent_len):
        svector = self.sentence_vector(sentence)
        emb = torch.sum(svector, 1)              # B x dim
        emb = emb / sent_len.expand_as(emb)
        return emb                               # B x dim

    def all_pooling(self, sentence, sent_len):
        return torch.cat([self.max_min_pooling(sentence, sent_len),
                self.average_pooling(sentence, sent_len)], axis=1)  # B x dim*3

    def score(self, groundtruth, t_len, generated, g_len):
        v1=self.pooling(groundtruth, t_len)
        v2=self.pooling(generated, g_len)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(v1, v2)




