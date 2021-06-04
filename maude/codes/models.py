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
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, DistilBertModel
from utils import get_optimizer
from sklearn.decomposition import PCA, IncrementalPCA
from collections import OrderedDict
from data import (
    ParlAIExtractor,
    DialogDataLoader,
    id_collate_fn,
    id_collate_nce_fn,
    context_collate_nce_fn,
)
import pytorch_lightning as pl
import pickle as pkl
from tqdm import tqdm
from logbook.logbook import LogBook
import copy
from codes.net import Net


class TransitionPredictorMaxPool(Net):
    def __init__(self, hparams, logbook=None):
        super(TransitionPredictorMaxPool, self).__init__(hparams, logbook)
        self.hparams = hparams
        self.copy_hparams = copy.deepcopy(self.hparams)
        self.train_data = None
        self.test_data = None
        self.down_model = None
        self.bert_input = True
        self.is_transition_fn = True
        dim = 0
        if "base-uncased" in hparams.bert_model:
            dim = 768
        else:
            raise NotImplementedError()
        with torch.no_grad():
            self.bert = self.init_bert_model()
        if hparams.downsample:
            if hparams.learn_down:
                self.down = nn.Parameter(torch.randn(dim, hparams.down_dim))
                nn.init.xavier_normal_(self.down)
                if hparams.fix_down:
                    self.down.requires_grad = False
            dim = hparams.down_dim
        self.lstm = nn.LSTM(
            dim,
            dim,
            2,
            bidirectional=hparams.bidirectional,
            batch_first=True,
            dropout=0,
        )
        if hparams.bidirectional:
            self.W = nn.Parameter(torch.randn(dim * 2, dim))
        else:
            self.W = nn.Parameter(torch.randn(dim, dim))
        # self.mpca = nn.Parameter(torch.randn(768, dim))
        self.decoder = nn.Sequential(
            nn.Linear(dim * 4, hparams.decoder_hidden),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.decoder_hidden, 1),
        )
        self.context_discriminator = nn.Sequential(
            nn.Linear(dim, hparams.decoder_hidden),
            nn.ReLU(),
            nn.Linear(hparams.decoder_hidden, 1),
        )
        nn.init.xavier_uniform_(self.W)
        self.init_bert_model()
        self.sigmoid = nn.Sigmoid()
        if self.hparams.train_mode == "nce":
            self.loss_fn = nn.BCELoss()
            self.collate_fn = id_collate_nce_fn
            if self.hparams.corrupt_type == "all_context":
                self.collate_fn = context_collate_nce_fn
        else:
            self.loss_fn = nn.MSELoss()
            self.collate_fn = id_collate_fn
        self.minibatch_step = 0

    def simple_pca_predict(self, tensor):
        """
        tensor: B x sent
        :param tensor:
        :return:
        """
        tokens = tensor.numpy()
        tokens = self.down_model.transform(tokens).astype(float)
        return tokens

    def downsample(self):
        """
        Downsample the data from BERT embeddings to lower dimensions
        :return:
        """
        if self.hparams.mode == "train" and not self.hparams.eval_val:
            # train pca
            self.pca_train()
            # self.save_pca_model()
        # predict tokens
        # self.dial_vecs = self.pca_predict(self.dial_vecs)

    def pca_train(self, vecs):
        """
        Train pca on the training split at the beginning and store the
        model in memory / persist
        :return:
        """
        self.logbook.write_message_logs("Training PCA ..")
        tokens = np.array(vecs)
        self.down_model = PCA(n_components=self.hparams.down_dim, whiten=True)
        self.down_model.fit(tokens)
        self.logbook.write_message_logs(
            "Saving PCA model in {}".format(self.hparams.pca_file)
        )
        pkl.dump({"pca": self.down_model}, open(self.hparams.pca_file, "wb"))

    def calc_ref_scores(self, batch):
        inp, inp_len, inp_dial_len, y_true, y_true_len, y_false, y_false_len = batch
        pred_true = self.forward(inp, inp_len, inp_dial_len, y_true, y_true_len)
        pred_false = self.forward(inp, inp_len, inp_dial_len, y_false, y_false_len)
        return pred_true, pred_false

    def calc_nce_scores(self, batch):
        inp, inp_len, inp_dial_len, y_true, y_true_len, y_falses, y_false_lens = batch
        pred_true = self.forward(inp, inp_len, inp_dial_len, y_true, y_true_len)
        num_neg_samples = y_falses.size(1)
        pred_falses = []
        for n in range(num_neg_samples):
            pred_falses.append(
                self.forward(
                    inp, inp_len, inp_dial_len, y_falses[:, n, :], y_false_lens[:, n, :]
                )
            )
        return pred_true, pred_falses

    def calc_nce_scores_batched(self, batch):
        inp, inp_len, inp_dial_len, y_true, y_true_len, y_falses, y_false_lens = batch
        num_neg_samples = y_falses.size(1)
        y_falses = y_falses.view(-1, y_falses.size(-1))
        y_false_lens = y_false_lens.view(-1, y_false_lens.size(-1))
        inp_false = torch.cat([inp for k in range(num_neg_samples)], dim=0)
        inp_len_false = torch.cat([inp_len for k in range(num_neg_samples)], dim=0)
        inp_dial_len_false = torch.cat(
            [inp_dial_len for k in range(num_neg_samples)], dim=0
        )
        pred_true = self.forward(inp, inp_len, inp_dial_len, y_true, y_true_len)
        pred_falses = self.forward(
            inp_false, inp_len_false, inp_dial_len_false, y_falses, y_false_lens
        )
        target_one = torch.ones_like(pred_true)
        target_zero = torch.zeros_like(pred_falses)
        target = torch.cat([target_one, target_zero], dim=0)
        return pred_true, pred_falses, target

    def calc_nce_scores_with_context_batched(self, batch):
        (
            inp,
            inp_len,
            inp_dial_len,
            y_true,
            y_true_len,
            y_falses,
            y_false_lens,
            inp_hat,
            inp_hat_len,
            inp_hat_dial_len,
        ) = batch
        num_neg_samples = y_falses.size(1)
        y_falses = y_falses.view(-1, y_falses.size(-1))
        y_false_lens = y_false_lens.view(-1, y_false_lens.size(-1))
        inp_false = torch.cat([inp for k in range(num_neg_samples)], dim=0)
        inp_len_false = torch.cat([inp_len for k in range(num_neg_samples)], dim=0)
        inp_dial_len_false = torch.cat(
            [inp_dial_len for k in range(num_neg_samples)], dim=0
        )
        pred_true = self.forward(inp, inp_len, inp_dial_len, y_true, y_true_len)
        pred_falses = self.forward(
            inp_false, inp_len_false, inp_dial_len_false, y_falses, y_false_lens
        )
        pred_inp_false = self.forward(
            inp_hat, inp_hat_len, inp_hat_dial_len, y_true, y_true_len
        )
        pred_falses = torch.cat([pred_falses, pred_inp_false], dim=0)
        target_one = torch.ones_like(pred_true)
        target_zero = torch.zeros_like(pred_falses)
        target = torch.cat([target_one, target_zero], dim=0)
        return pred_true, pred_falses, target

    def forward(
        self,
        dials,
        dial_length,
        dial_word_len,
        response,
        response_len,
        mode="ref_score",
    ):
        # import ipdb; ipdb.set_trace()
        dtype = self.lstm.weight_ih_l0.dtype
        device = dials.device  # self.trainer.proc_rank
        batch, num_dial, num_words = dials.shape
        dials = dials.view(-1, num_words)
        dials = self.extract_sentence_bert(dials, dial_word_len)
        dials = dials.view(batch, num_dial, -1)
        batch, num_dial, dim = dials.shape
        if self.hparams.downsample:
            if self.hparams.learn_down:
                dials = torch.matmul(dials.view(-1, dim), self.down)
                dials = dials.view(batch, num_dial, -1)
            else:
                dials = self.simple_pca_predict(dials.view(-1, dim).to("cpu"))
                # dials = torch.bmm(dials, self.mpca.unsqueeze(0).repeat(dials.size(0), 1, 1))
                dials = torch.tensor(dials, dtype=dtype, device=device).view(
                    batch, num_dial, -1
                )
        # inp_vec = [self.data.pca_predict([self.extract_sentence_bert(batch)])[0] for batch in dials]
        # inp_vec, _ = batchify(inp_vec, vector_mode=True)
        # dials = dials.to(device)
        response = response.unsqueeze(1)  # B x 1 x s
        response = self.extract_sentence_bert(response, response_len)
        # response = torch.bmm(response, self.mpca.unsqueeze(0).repeat(response.size(0), 1, 1))
        if self.hparams.downsample:
            if self.hparams.learn_down:
                response = torch.matmul(response.squeeze(1), self.down).unsqueeze(1)
            else:
                response = self.simple_pca_predict(response.squeeze(1).to("cpu"))
                response = torch.tensor(response, dtype=dtype, device=device)
        # response = response.to(device)

        # convert to proper floats
        # dtype = self.lstm.weight_ih_l0.dtype
        # dials = dials.to(dtype)
        # response = response.to(dtype)
        dial_length = dial_length.cpu().numpy()  # np.array(dial_length)
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
        # import ipdb; ipdb.set_trace()
        outp, hidden_rep = self.lstm(data_pack)
        outp, _ = pad_packed_sequence(outp, batch_first=True)
        outp = outp.contiguous()
        outp_l = torch.zeros(
            (dials.size(0), dials.size(1), outp.size(2)), dtype=dtype, device=device
        )
        # import ipdb; ipdb.set_trace()
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
        ref_score = self.sigmoid(self.decoder(rep))
        context_score = self.sigmoid(self.context_discriminator(hidden_rep))
        if mode == "ref_score":
            return ref_score
        else:
            return context_score

    def preflight_steps(self):
        """
        Extract all training BERT embeddings and train pca
        Do it only if we do not have a saved file
        :return:
        """
        if not self.hparams.learn_down and not self.hparams.fix_down:
            self.logbook.write_message_logs(
                "Checking pca file in ... {}".format(self.hparams.pca_file)
            )
            if not self.down_model:
                if os.path.exists(self.hparams.pca_file) and os.path.isfile(
                    self.hparams.pca_file
                ):
                    self.logbook.write_message_logs(
                        "Loading PCA model from {}".format(self.hparams.pca_file)
                    )
                    data_dump = pkl.load(open(self.hparams.pca_file, "rb"))
                    self.down_model = data_dump["pca"]
                else:
                    self.logbook.write_message_logs(
                        "Not found. Proceeding to extract and train..."
                    )
                    self.down_model = IncrementalPCA(
                        n_components=self.hparams.down_dim, whiten=True
                    )
                    # extract and save embeddings
                    train_loader = self.get_dataloader(mode="train")
                    all_vecs = []
                    self.logbook.write_message_logs("Extracting embeddings ...")
                    pb = tqdm(total=len(train_loader))
                    for bi, batch in enumerate(train_loader):
                        (
                            inp,
                            inp_len,
                            inp_dial_len,
                            y_true,
                            y_true_len,
                            y_false,
                            y_false_len,
                        ) = batch
                        if inp.size(0) < self.hparams.batch_size:
                            continue
                        with torch.no_grad():
                            batch, num_dials, num_words = inp.shape
                            inp = inp.view(-1, num_words).to(self.hparams.device)
                            inp_dial_len = inp_dial_len.to(self.hparams.device)
                            inp_vec = self.extract_sentence_bert(inp, inp_dial_len)
                            inp_vec = inp_vec.view(batch, num_dials, -1)  # B x D x dim
                            inp_vec = (
                                inp_vec.view(-1, inp_vec.size(2)).to("cpu").numpy()
                            )  # (B x D) x dim
                            self.down_model.partial_fit(inp_vec)
                        del inp
                        del inp_len
                        del inp_vec
                        del y_true
                        del y_false
                        # temporary solution...
                        torch.cuda.empty_cache()
                        pb.update(1)
                        # if bi == 100:
                        #     break
                    pb.close()
                    self.logbook.write_message_logs(
                        "Saving PCA model in {}".format(self.hparams.pca_file)
                    )
                    pkl.dump(
                        {"pca": self.down_model}, open(self.hparams.pca_file, "wb")
                    )
