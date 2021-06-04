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
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# TODO: this is non-batched. make a batched implementation
class TransitionFn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dim = 0
        if args.bert_model == "bert-base-uncased":
            dim = 768
        else:
            raise NotImplementedError()
        if args.downsample:
            dim = args.down_dim
        self.lstm = nn.LSTM(
            dim, dim, 1, bidirectional=False, batch_first=True, dropout=0
        )

    def forward(self, dials):
        outp, hidden_rep = self.lstm(dials)
        return hidden_rep[0]


class TransitionPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dim = 0
        if args.bert_model == "bert-base-uncased":
            dim = 768
        else:
            raise NotImplementedError()
        if args.downsample:
            dim = args.down_dim
        self.lstm = nn.LSTM(
            dim, dim, 1, bidirectional=False, batch_first=True, dropout=0
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim * 2, 100), nn.ReLU(), nn.Linear(100, 1)
        )
        self.W = nn.Parameter(torch.randn(dim, dim))
        nn.init.xavier_uniform_(self.W)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dials, dial_length, response):
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
        hidden_rep = outp_l[:, -1, :]
        hidden_rep = torch.bmm(
            hidden_rep.unsqueeze(1),
            self.W.unsqueeze(0).repeat(hidden_rep.size(0), 1, 1),
        )
        rep = torch.cat([hidden_rep.squeeze(1), response.squeeze(1)], dim=1)
        return self.sigmoid(self.decoder(rep))


class TransitionPredictorIS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dim = 0
        if args.bert_model == "bert-base-uncased":
            dim = 768
        else:
            raise NotImplementedError()
        if args.downsample:
            dim = args.down_dim
        self.lstm = nn.LSTM(
            dim, dim, 1, bidirectional=False, batch_first=True, dropout=0
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim * 4, 200), nn.ReLU(), nn.Linear(200, 1)
        )
        self.W = nn.Parameter(torch.randn(dim, dim))
        nn.init.xavier_uniform_(self.W)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dials, dial_length, response):
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
        hidden_rep = outp_l[:, -1, :]
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


class TransitionRotation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dim = 0
        if args.bert_model == "bert-base-uncased":
            dim = 768
        else:
            raise NotImplementedError()
        if args.downsample:
            dim = args.down_dim
        self.lstm = nn.LSTM(
            dim, dim, 1, bidirectional=False, batch_first=True, dropout=0
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim * 2 + 1, 100), nn.ReLU(), nn.Linear(100, 1)
        )
        self.W = nn.Parameter(torch.randn(dim, dim))
        nn.init.xavier_uniform_(self.W)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dials, response):
        inp = dials
        outp, hidden_rep = self.lstm(inp)
        xW = torch.matmul(hidden_rep[0].squeeze(0), self.W)
        xWy = torch.matmul(xW, response.transpose(0, 1))
        out = torch.cat([hidden_rep[0].squeeze(0), xWy, response], dim=1)
        return self.sigmoid(out)


class TransitionPredictorCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dim = 0
        if args.bert_model == "bert-base-uncased":
            dim = 768
        else:
            raise NotImplementedError()
        if args.downsample:
            dim = args.down_dim
        self.lstm = nn.LSTM(
            dim, dim, 1, bidirectional=False, batch_first=True, dropout=0
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim * 2, 100), nn.ReLU(), nn.Linear(100, 1)
        )
        self.cnn = nn.Conv1d()
        self.sigmoid = nn.Sigmoid()

    def forward(self, dials, response):
        outp, hidden_rep = self.lstm(dials)
        a = hidden_rep[0].squeeze(1)
        b = response
        rep = a[:, :, None] @ b[:, None, :]

        return self.sigmoid(self.decoder(rep))
