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
from transformers import BertModel
from data import (
    id_collate_flat_fn,
    id_collate_flat_nce_fn,
    context_collate_flat_fn,
    context_collate_flat_nce_fn,
)
import copy
from backup.baselines.ruber.model import BiLSTMEncoder, get_mlp
from codes.net import Net


class RuberUnreferenced(Net):
    def __init__(self, hparams, logbook=None):
        super(RuberUnreferenced, self).__init__(hparams, logbook)
        self.hparams = hparams
        self.copy_hparams = copy.deepcopy(self.hparams)
        self.minibatch_step = 0
        self.word_dict = None
        self.emb = nn.Embedding(
            hparams.num_words, hparams.word2vec_embedding_dim, padding_idx=0
        )
        # if hparams.ruber_load_emb:
        #    self.emb.weight = torch.load(hparams.word2vec_out)
        self.queryGRU = BiLSTMEncoder(hparams, self.emb)
        self.replyGRU = BiLSTMEncoder(hparams, self.emb)
        self.quadratic_M = nn.Parameter(
            torch.zeros(
                (hparams.word2vec_embedding_dim * 2, hparams.word2vec_embedding_dim * 2)
            )
        )
        nn.init.zeros_(self.quadratic_M)

        # self.mlp = get_mlp(
        #     (hparams.word2vec_embedding_dim * 4 + 1),
        #     1,
        #     2,
        #     hidden_dim=hparams.ruber_mlp_dim,
        # )
        self.mlp = nn.Sequential(
            nn.Linear(hparams.word2vec_embedding_dim * 4 + 1, hparams.ruber_mlp_dim),
            nn.ReLU(),
            nn.Linear(hparams.ruber_mlp_dim, 1),
        )
        self.train_data = None
        self.test_data = None
        self.loss_fn = nn.MSELoss()
        if self.hparams.train_mode == "nce":
            self.collate_fn = id_collate_flat_nce_fn
            if self.hparams.corrupt_type == "all_context":
                self.collate_fn = context_collate_flat_nce_fn
        else:
            self.collate_fn = id_collate_flat_fn

    def forward(self, query_batch, query_length, reply_batch, reply_length):
        qout = self.queryGRU((query_batch.transpose(1, 0), query_length))  # B x dim * 2
        rout = self.replyGRU((reply_batch.transpose(1, 0), reply_length))  # B x dim * 2

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

    def preflight_steps(self):
        """
        Extract all training BERT embeddings and train pca
        Do it only if we do not have a saved file
        :return:
        """
        self.logbook.write_message_logs(
            "Checking word2vec file in ... {}".format(self.hparams.word2vec_out)
        )
        if os.path.exists(self.hparams.word2vec_out) and os.path.isfile(
            self.hparams.word2vec_out
        ):
            self.logbook.write_message_logs("Word2vec file present")
        else:
            raise NotImplementedError("need pre-trained word2vec file")

    def calc_ref_scores(self, batch):
        inp, inp_len, y_true, y_len, y_false, y_false_len = batch
        pred_true = self.forward(inp, inp_len, y_true, y_len)
        pred_false = self.forward(inp, inp_len, y_false, y_false_len)
        return pred_true, pred_false

    def calc_nce_scores(self, batch):
        inp, inp_len, y_true, y_true_len, y_falses, y_false_lens = batch
        pred_true = self.forward(inp, inp_len, y_true, y_true_len)
        num_neg_samples = y_falses.size(1)
        pred_falses = []
        for n in range(num_neg_samples):
            neg_inp = y_falses[:, n, :]
            neg_len = y_false_lens[:, n, :]
            if neg_len.dim() > 1:
                neg_len = neg_len.squeeze()
            pred_falses.append(self.forward(inp, inp_len, neg_inp, neg_len))
        return pred_true, pred_falses

    def calc_nce_scores_batched(self, batch):
        inp, inp_len, y_true, y_true_len, y_falses, y_false_lens = batch
        num_neg_samples = y_falses.size(1)
        y_falses = y_falses.view(-1, y_falses.size(-1))
        y_false_lens = y_false_lens.view(-1, y_false_lens.size(-1))
        inp_false = torch.cat([inp for k in range(num_neg_samples)], dim=0)
        inp_len_false = torch.cat([inp_len for k in range(num_neg_samples)], dim=0)
        pred_true = self.forward(inp, inp_len, y_true, y_true_len)
        if y_false_lens.dim() > 1:
            y_false_lens = y_false_lens.squeeze()
        pred_falses = self.forward(inp_false, inp_len_false, y_falses, y_false_lens)
        target_one = torch.ones_like(pred_true)
        target_zero = torch.zeros_like(pred_falses)
        target = torch.cat([target_one, target_zero], dim=0)
        return pred_true, pred_falses, target

    def calc_cont_scores(self, batch):
        raise NotImplementedError("context discriminator not implemented")


class InferSent(Net):
    def __init__(self, hparams, logbook=None):
        super(InferSent, self).__init__(hparams, logbook)
        self.hparams = hparams
        self.copy_hparams = copy.deepcopy(self.hparams)
        self.minibatch_step = 0
        self.word_dict = None
        self.emb = nn.Embedding(
            hparams.num_words, hparams.word2vec_embedding_dim, padding_idx=0
        )
        # if hparams.ruber_load_emb:
        #    self.emb.weight = torch.load(hparams.word2vec_out)
        self.queryGRU = BiLSTMEncoder(hparams, self.emb)
        self.replyGRU = BiLSTMEncoder(hparams, self.emb)
        self.W = nn.Parameter(
            torch.zeros(
                (hparams.word2vec_embedding_dim * 2, hparams.word2vec_embedding_dim * 2)
            )
        )
        nn.init.xavier_uniform_(self.W)

        self.decoder = nn.Sequential(
            nn.Linear(hparams.word2vec_embedding_dim * 8, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )
        self.loss_fn = nn.MSELoss()
        self.train_data = None
        self.test_data = None
        if self.hparams.train_mode == "nce":
            self.collate_fn = id_collate_flat_nce_fn
            if self.hparams.corrupt_type == "all_context":
                self.collate_fn = context_collate_flat_nce_fn
        else:
            self.collate_fn = id_collate_flat_fn

    def forward(self, query_batch, query_length, reply_batch, reply_length):
        qout = self.queryGRU((query_batch.transpose(1, 0), query_length))  # B x dim * 2
        rout = self.replyGRU((reply_batch.transpose(1, 0), reply_length))  # B x dim * 2

        qout = torch.bmm(
            qout.unsqueeze(1), self.W.unsqueeze(0).repeat(qout.size(0), 1, 1)
        )
        qout = qout.squeeze(1)
        rep = torch.cat([qout, rout, torch.abs(qout - rout), qout * rout], dim=1)
        ref_score = torch.sigmoid(self.decoder(rep))
        # out = B x 1
        return ref_score

    def calc_ref_scores(self, batch):
        inp, inp_len, y_true, y_len, y_false, y_false_len = batch
        pred_true = self.forward(inp, inp_len, y_true, y_len)
        pred_false = self.forward(inp, inp_len, y_false, y_false_len)
        return pred_true, pred_false

    def calc_nce_scores(self, batch):
        inp, inp_len, y_true, y_true_len, y_falses, y_false_lens = batch
        pred_true = self.forward(inp, inp_len, y_true, y_true_len)
        num_neg_samples = y_falses.size(1)
        pred_falses = []
        for n in range(num_neg_samples):
            pred_falses.append(
                self.forward(
                    inp, inp_len, y_falses[:, n, :], y_false_lens[:, n, :].squeeze()
                )
            )
        return pred_true, pred_falses

    def calc_nce_scores_batched(self, batch):
        inp, inp_len, y_true, y_true_len, y_falses, y_false_lens = batch
        num_neg_samples = y_falses.size(1)
        y_falses = y_falses.view(-1, y_falses.size(-1))
        y_false_lens = y_false_lens.view(-1, y_false_lens.size(-1))
        inp_false = torch.cat([inp for k in range(num_neg_samples)], dim=0)
        inp_len_false = torch.cat([inp_len for k in range(num_neg_samples)], dim=0)
        pred_true = self.forward(inp, inp_len, y_true, y_true_len)
        if y_false_lens.dim() > 1:
            y_false_lens = y_false_lens.squeeze()
        pred_falses = self.forward(inp_false, inp_len_false, y_falses, y_false_lens)
        target_one = torch.ones_like(pred_true)
        target_zero = torch.zeros_like(pred_falses)
        target = torch.cat([target_one, target_zero], dim=0)
        return pred_true, pred_falses, target

    def calc_nce_scores_with_context_batched(self, batch):
        (
            inp,
            inp_len,
            y_true,
            y_true_len,
            y_falses,
            y_false_lens,
            inp_hat,
            inp_hat_len,
        ) = batch
        num_neg_samples = y_falses.size(1)
        y_falses = y_falses.view(-1, y_falses.size(-1))
        y_false_lens = y_false_lens.view(-1, y_false_lens.size(-1))
        inp_false = torch.cat([inp for k in range(num_neg_samples)], dim=0)
        inp_len_false = torch.cat([inp_len for k in range(num_neg_samples)], dim=0)
        pred_true = self.forward(inp, inp_len, y_true, y_true_len)
        if y_false_lens.dim() > 1:
            y_false_lens = y_false_lens.squeeze()
        pred_falses = self.forward(inp_false, inp_len_false, y_falses, y_false_lens)
        pred_inp_false = self.forward(inp_hat, inp_hat_len, y_true, y_true_len)
        pred_falses = torch.cat([pred_falses, pred_inp_false], dim=0)
        target_one = torch.ones_like(pred_true)
        target_zero = torch.zeros_like(pred_falses)
        target = torch.cat([target_one, target_zero], dim=0)
        return pred_true, pred_falses, target

    def calc_cont_scores(self, batch):
        raise NotImplementedError("context discriminator not implemented")

    def preflight_steps(self):
        """
        Extract all training BERT embeddings and train pca
        Do it only if we do not have a saved file
        :return:
        """
        self.logbook.write_message_logs(
            "Checking word2vec file in ... {}".format(self.hparams.word2vec_out)
        )
        if os.path.exists(self.hparams.word2vec_out) and os.path.isfile(
            self.hparams.word2vec_out
        ):
            self.logbook.write_message_logs("Word2vec file present")
        else:
            raise NotImplementedError("need pre-trained word2vec file")


class BERTNLI(Net):
    def __init__(self, hparams, logbook=None):
        super(BERTNLI, self).__init__(hparams, logbook)
        self.hparams = hparams
        self.copy_hparams = copy.deepcopy(self.hparams)
        self.minibatch_step = 0
        self.bert_input = True
        if "base-uncased" in hparams.bert_model:
            dim = 768
        else:
            raise NotImplementedError()
        self.bert = self.init_bert_model()
        self.W = nn.Parameter(torch.zeros((dim, dim)))
        nn.init.xavier_uniform_(self.W)
        self.dim = dim

        self.decoder = nn.Sequential(
            nn.Linear(dim * 4, hparams.decoder_hidden),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.decoder_hidden, 1),
        )
        self.loss_fn = nn.MSELoss()
        self.train_data = None
        self.test_data = None
        if self.hparams.train_mode == "nce":
            self.collate_fn = id_collate_flat_nce_fn
            if self.hparams.corrupt_type == "all_context":
                self.collate_fn = context_collate_flat_nce_fn
        else:
            self.collate_fn = id_collate_flat_fn

    def forward(self, query_batch, query_length, reply_batch, reply_length):
        qout = self.extract_sentence_bert(query_batch, query_length)  # B x dim
        rout = self.extract_sentence_bert(reply_batch, reply_length)  # B x dim
        # assert query_batch.max().item() < 30522
        # assert query_batch.min().item() == 0
        # assert reply_batch.max().item() < 30522
        # assert reply_batch.min().item() == 0
        qout = torch.bmm(
            qout.unsqueeze(1), self.W.unsqueeze(0).repeat(qout.size(0), 1, 1)
        )
        qout = qout.squeeze(1)
        rep = torch.cat([qout, rout, torch.abs(qout - rout), qout * rout], dim=1)
        ref_score = torch.sigmoid(self.decoder(rep))
        # out = B x 1
        return ref_score

    def calc_ref_scores(self, batch):
        inp, inp_len, y_true, y_len, y_false, y_false_len = batch
        pred_true = self.forward(inp, inp_len, y_true, y_len)
        pred_false = self.forward(inp, inp_len, y_false, y_false_len)
        return pred_true, pred_false

    def calc_nce_scores(self, batch):
        inp, inp_len, y_true, y_true_len, y_falses, y_false_lens = batch
        pred_true = self.forward(inp, inp_len, y_true, y_true_len)
        num_neg_samples = y_falses.size(1)
        pred_falses = []
        for n in range(num_neg_samples):
            pred_falses.append(
                self.forward(inp, inp_len, y_falses[:, n, :], y_false_lens[:, n, :])
            )
        return pred_true, pred_falses

    def calc_nce_scores_batched(self, batch):
        inp, inp_len, y_true, y_true_len, y_falses, y_false_lens = batch
        num_neg_samples = y_falses.size(1)
        y_falses = y_falses.view(-1, y_falses.size(-1))
        y_false_lens = y_false_lens.view(-1, y_false_lens.size(-1))
        inp_false = torch.cat([inp for k in range(num_neg_samples)], dim=0)
        inp_len_false = torch.cat([inp_len for k in range(num_neg_samples)], dim=0)
        pred_true = self.forward(inp, inp_len, y_true, y_true_len)
        pred_falses = self.forward(inp_false, inp_len_false, y_falses, y_false_lens)
        target_one = torch.ones_like(pred_true)
        target_zero = torch.zeros_like(pred_falses)
        target = torch.cat([target_one, target_zero], dim=0)
        return pred_true, pred_falses, target

    def calc_nce_scores_with_context_batched(self, batch):
        (
            inp,
            inp_len,
            y_true,
            y_true_len,
            y_falses,
            y_false_lens,
            inp_hat,
            inp_hat_len,
        ) = batch
        num_neg_samples = y_falses.size(1)
        y_falses = y_falses.view(-1, y_falses.size(-1))
        y_false_lens = y_false_lens.view(-1, y_false_lens.size(-1))
        inp_false = torch.cat([inp for k in range(num_neg_samples)], dim=0)
        inp_len_false = torch.cat([inp_len for k in range(num_neg_samples)], dim=0)
        pred_true = self.forward(inp, inp_len, y_true, y_true_len)
        pred_falses = self.forward(inp_false, inp_len_false, y_falses, y_false_lens)
        pred_inp_false = self.forward(inp_hat, inp_hat_len, y_true, y_true_len)
        pred_falses = torch.cat([pred_falses, pred_inp_false], dim=0)
        target_one = torch.ones_like(pred_true)
        target_zero = torch.zeros_like(pred_falses)
        target = torch.cat([target_one, target_zero], dim=0)
        return pred_true, pred_falses, target

    def calc_cont_scores(self, batch):
        raise NotImplementedError("context discriminator not implemented")
