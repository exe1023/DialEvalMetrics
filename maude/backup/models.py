"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# Metric training models
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_transformers import BertModel
from utils import batchify


class BaseModel(nn.Module):
    def __init__(self, args, data, logbook):
        super().__init__()
        self.args = args
        self.logbook = logbook
        self.data = data
        self.init_bert_model()

    def init_bert_model(self):
        """
        Initialize bert pretrained or finetuned model
        :return:
        """
        load_path = self.args.bert_model
        if self.args.load_fine_tuned:
            load_path = "fine_tune_{}_{}/".format(
                self.args.data_name, self.args.trained_bert_suffix
            )
        self.logbook.write_message_logs("Loading bert from {}".format(load_path))
        self.bert = BertModel.from_pretrained(load_path)
        self.bert.to(self.args.device)
        self.bert.eval()

    def extract_sentence_bert(self, sents):
        """
        Extract sentence bert representation
        where sents is a batch of sentences
        :param sent:
        :return:
        """
        max_indx_len = max([len(sent) for sent in sents])
        segments_tensor = (
            torch.zeros(len(sents), max_indx_len).long().to(self.args.device)
        )
        # batch indexes
        tokens_tensor = (
            torch.zeros(len(sents), max_indx_len).long().to(self.args.device)
        )
        for i, indx in enumerate(sents):
            tokens_tensor[i][: len(indx)] = torch.LongTensor(indx)
        with torch.no_grad():
            outs = self.bert(tokens_tensor, token_type_ids=segments_tensor)
            return outs[1].to("cpu")

    def forward(self, X, X_len, Y):
        pass


class TransitionPredictorMaxPool(BaseModel):
    def __init__(self, args, data, logbook):
        super().__init__(args, data, logbook)
        dim = 0
        if args.bert_model == "bert-base-uncased":
            dim = 768
        else:
            raise NotImplementedError()
        if args.downsample:
            dim = args.down_dim
        self.lstm = nn.LSTM(
            dim, dim, 2, bidirectional=args.bidirectional, batch_first=True, dropout=0
        )
        if args.bidirectional:
            self.W = nn.Parameter(torch.randn(dim * 2, dim))
        else:
            self.W = nn.Parameter(torch.randn(dim, dim))
        self.decoder = nn.Sequential(
            nn.Linear(dim * 4, 200), nn.ReLU(), nn.Linear(200, 1)
        )
        nn.init.xavier_uniform_(self.W)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dials, dial_length, response):
        inp_vec = [
            self.data.pca_predict([self.extract_sentence_bert(batch)])[0]
            for batch in dials
        ]
        inp_vec, _ = batchify(inp_vec, vector_mode=True)
        dials = inp_vec.to(self.args.device)
        y = self.data.pca_predict([self.extract_sentence_bert(response)])[0]
        response = torch.stack(y, dim=0)
        response = response.to(self.args.device)
        dial_length = np.array(dial_length)
        inp_len_sorted, idx_sort = np.sort(dial_length)[::-1], np.argsort(-dial_length)
        inp_len_sorted = inp_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).to(dials.device)
        dials = dials.index_select(0, idx_sort)
        inp_len_sorted_nonzero_idx = np.nonzero(inp_len_sorted)[0]
        inp_len_sorted_nonzero_idx = torch.from_numpy(inp_len_sorted_nonzero_idx).to(
            dials.device
        )
        inp_len_sorted = torch.from_numpy(inp_len_sorted).to(dials.device)
        non_zero_data = dials.index_select(0, inp_len_sorted_nonzero_idx)
        data_pack = pack_padded_sequence(
            non_zero_data, inp_len_sorted[inp_len_sorted_nonzero_idx], batch_first=True
        )
        outp, hidden_rep = self.lstm(data_pack)
        outp, _ = pad_packed_sequence(outp, batch_first=True)
        outp = outp.contiguous()
        outp_l = torch.zeros((dials.size(0), dials.size(1), outp.size(2))).to(
            outp.device
        )
        outp_l[inp_len_sorted_nonzero_idx] = outp
        # unsort
        idx_unsort = torch.from_numpy(idx_unsort).to(outp_l.device)
        outp_l = outp_l.index_select(0, idx_unsort)

        # last outp
        hidden_rep = torch.max(outp_l, 1)[0]
        hidden_rep = torch.bmm(
            hidden_rep.unsqueeze(1),
            self.W.unsqueeze(0).repeat(hidden_rep.size(0), 1, 1),
        )
        hidden_rep = hidden_rep.squeeze(1)
        response = response.squeeze(1)
        rep = torch.cat(
            [
                hidden_rep,
                response,
                torch.abs(hidden_rep - response),
                hidden_rep * response,
            ],
            dim=1,
        )
        return self.sigmoid(self.decoder(rep))


class TransitionPredictorMaxPoolLearnedDownsample(BaseModel):
    def __init__(self, args, data, logbook):
        super().__init__(args, data, logbook)
        dim = 0
        if args.bert_model == "bert-base-uncased":
            dim = 768
        else:
            raise NotImplementedError()
        if args.downsample:
            self.down = nn.Parameter(torch.randn(dim, args.down_dim))
            dim = args.down_dim
        self.lstm = nn.LSTM(
            dim, dim, 2, bidirectional=args.bidirectional, batch_first=True, dropout=0
        )
        if args.bidirectional:
            self.W = nn.Parameter(torch.randn(dim * 2, dim))
        else:
            self.W = nn.Parameter(torch.randn(dim, dim))
        self.decoder = nn.Sequential(
            nn.Linear(dim * 4, 200), nn.ReLU(), nn.Linear(200, 1)
        )
        nn.init.xavier_uniform_(self.W)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dials, dial_length, response):
        import ipdb

        ipdb.set_trace()
        inp_vec = torch.stack(
            [self.extract_sentence_bert(batch) for batch in dials], dim=0
        )
        inp_vec = torch.mul(inp_vec, self.down)
        inp_vec, _ = batchify(inp_vec, vector_mode=True)
        dials = inp_vec.to(self.args.device)
        y = self.data.pca_predict([self.extract_sentence_bert(response)])[0]
        response = torch.stack(y, dim=0)
        response = response.to(self.args.device)
        dial_length = np.array(dial_length)
        inp_len_sorted, idx_sort = np.sort(dial_length)[::-1], np.argsort(-dial_length)
        inp_len_sorted = inp_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).to(dials.device)
        dials = dials.index_select(0, idx_sort)
        inp_len_sorted_nonzero_idx = np.nonzero(inp_len_sorted)[0]
        inp_len_sorted_nonzero_idx = torch.from_numpy(inp_len_sorted_nonzero_idx).to(
            dials.device
        )
        inp_len_sorted = torch.from_numpy(inp_len_sorted).to(dials.device)
        non_zero_data = dials.index_select(0, inp_len_sorted_nonzero_idx)
        data_pack = pack_padded_sequence(
            non_zero_data, inp_len_sorted[inp_len_sorted_nonzero_idx], batch_first=True
        )
        outp, hidden_rep = self.lstm(data_pack)
        outp, _ = pad_packed_sequence(outp, batch_first=True)
        outp = outp.contiguous()
        outp_l = torch.zeros((dials.size(0), dials.size(1), outp.size(2))).to(
            outp.device
        )
        outp_l[inp_len_sorted_nonzero_idx] = outp
        # unsort
        idx_unsort = torch.from_numpy(idx_unsort).to(outp_l.device)
        outp_l = outp_l.index_select(0, idx_unsort)

        # last outp
        hidden_rep = torch.max(outp_l, 1)[0]
        hidden_rep = torch.bmm(
            hidden_rep.unsqueeze(1),
            self.W.unsqueeze(0).repeat(hidden_rep.size(0), 1, 1),
        )
        hidden_rep = hidden_rep.squeeze(1)
        response = response.squeeze(1)
        rep = torch.cat(
            [
                hidden_rep,
                response,
                torch.abs(hidden_rep - response),
                hidden_rep * response,
            ],
            dim=1,
        )
        return self.sigmoid(self.decoder(rep))
