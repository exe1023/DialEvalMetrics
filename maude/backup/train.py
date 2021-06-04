"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# Training automatic dialog evaluation metric on top of BERT embeddings
import torch
import torch.nn as nn
from utils import get_optimizer, batch, batchify, _import_module, BaseTrainer
from data import Data, ParlAIExtractor
import numpy as np
import random
from args import get_args
from logbook.logbook import LogBook
import os


class DialogMetric(BaseTrainer):
    def __init__(self, args, data, logbook):
        super().__init__(args, data, logbook)
        if args.downsample:
            self.logbook.write_message_logs("Downsampling to {}".format(args.down_dim))
            self.data.downsample()
        model = _import_module(args.model)(self.args, data, logbook)
        self.setup_model_opt(model)
        self.loss_fn = nn.MSELoss()
        # self.init_bert_model()
        # enable this if you want to watch gradient and parameters
        if args.watch_model:
            self.logbook.watch_model(self.model)

    def train_epoch(self, epoch=0):
        """
        Train a binary predictor
        :return:
        """
        self.model.train()
        # X, Y, Y_hat = self.data.prepare_data(self.data.train_indices, self.args.vector_mode)
        loss_a = []
        t_loss_a = []
        f_loss_a = []
        # num_batches = len(list(range(0, len(X), self.args.batch_size)))
        dl = self.data.get_dataloader(mode="train", epoch=epoch)
        for i, batch in enumerate(dl):
            # for i in range(0, len(X), self.args.batch_size):
            # inp_vec, inp_len = batchify(X[i:i+self.args.batch_size], self.args.vector_mode)
            # outp_vec, outp_len = batchify(Y[i:i+self.args.batch_size], self.args.vector_mode)
            inp, inp_len, y_true, y_false = batch
            # inp_vec = [self.data.pca_predict([self.extract_sentence_bert(batch)])[0] for batch in inp]
            # inp_vec, _ = batchify(inp_vec, vector_mode=True)
            # inp_vec = inp_vec.to(self.device)
            # y_true = self.data.pca_predict([self.extract_sentence_bert(y_true)])[0]
            # y_true = torch.stack(y_true, dim=0)
            # y_true = y_true.to(self.device)
            # y_false = self.data.pca_predict([self.extract_sentence_bert(y_false)])[0]
            # y_false = torch.stack(y_false, dim=0)
            # y_false = y_false.to(self.device)
            pred_true = self.model(inp, inp_len, y_true)
            pred_false = self.model(inp, inp_len, y_false)
            # import pdb; pdb.set_trace()
            t_loss = self.loss_fn(
                pred_true, torch.ones(pred_true.size(0), 1).to(self.device)
            )
            f_loss = self.loss_fn(
                pred_false, torch.zeros(pred_false.size(0), 1).to(self.device)
            )
            loss = t_loss + f_loss
            self.optimizer.zero_grad()
            loss.backward()
            loss_a.append(loss.item())
            t_loss_a.append(t_loss.item())
            f_loss_a.append(f_loss.item())
            if i % self.args.log_interval == 0:
                metrics = {
                    "mode": "train",
                    "minibatch": self.train_step,
                    "loss": np.mean(loss_a),
                    "true_loss": np.mean(t_loss_a),
                    "false_loss": np.mean(f_loss_a),
                    "epoch": epoch,
                }
                self.train_step += 1
                loss_a = []
                t_loss_a = []
                f_loss_a = []
                self.logbook.write_metric_logs(metrics)
            self.optimizer.step()
        # post epoch
        metrics = {
            "mode": "train",
            "minibatch": self.train_step,
            "loss": np.mean(loss_a),
            "true_loss": np.mean(t_loss_a),
            "false_loss": np.mean(f_loss_a),
            "epoch": epoch,
        }
        self.train_step += 1
        self.logbook.write_metric_logs(metrics)

    def evaluate_epoch(self, epoch=0, mode="valid"):
        """
        Eval binary trained metric
        :return:
        """
        self.model.eval()
        # X, Y, Y_hat = self.data.prepare_data(self.data.test_indices, self.args.vector_mode)
        test_scores = []
        test_scores_s = []
        # num_batches = len(list(range(0, len(X), self.args.batch_size)))
        dl = self.data.get_dataloader(mode="test", epoch=epoch)
        for i, batch in enumerate(dl):
            # for i in range(0, len(X), self.args.batch_size):
            # inp_vec, inp_len = batchify(X[i:i + self.args.batch_size], self.args.vector_mode)
            # outp_vec, outp_len = batchify(Y[i:i + self.args.batch_size], self.args.vector_mode)

            # inp_vec = inp_vec.to(self.device)
            # outp_vec = outp_vec.to(self.device)
            inp, inp_len, y_true, y_false = batch
            # inp_vec = [self.data.pca_predict([self.extract_sentence_bert(batch)])[0] for batch in inp]
            # inp_vec, _ = batchify(inp_vec, vector_mode=True)
            # inp_vec = inp_vec.to(self.device)
            # y_true = self.data.pca_predict([self.extract_sentence_bert(y_true)])[0]
            # y_true = torch.stack(y_true, dim=0)
            # y_true = y_true.to(self.device)
            # y_false = self.data.pca_predict([self.extract_sentence_bert(y_false)])[0]
            # y_false = torch.stack(y_false, dim=0)
            # y_false = y_false.to(self.args.device)
            pred_true = self.model(inp, inp_len, y_true)
            pred_false = self.model(inp, inp_len, y_false)
            test_scores.append(pred_true.mean().item())
            test_scores_s.append(pred_false.mean().item())

        metrics = {
            "mode": mode,
            "minibatch": self.train_step,
            "epoch": epoch,
            "score_mean": np.mean(test_scores),
            "score_std": np.std(test_scores),
            "scramble_score_mean": np.mean(test_scores_s),
            "scramble_score_std": np.std(test_scores_s),
            "mean_diff": np.mean(test_scores) - np.mean(test_scores_s),
        }
        self.logbook.write_metric_logs(metrics)

        # print("Test, Mean : {}, Std : {}".format(np.mean(test_scores), np.std(test_scores)))
        # print("Test Scrambled, Mean : {}, Std : {}".format(np.mean(test_scores_s), np.std(test_scores_s)))
        # print("Difference mean : {}".format(np.mean(test_scores_s) - np.mean(test_scores)))

    # def train_likelihood_epoch(self):
    #     """
    #     Train a simple metric to predict the BERT embeddings
    #     :return:
    #     """
    #     self.model.train()
    #     X,Y,Y_hat = self.data.prepare_data(self.data.train_indices)
    #     for i in range(len(X)):
    #         inp_vec = batch(X[i]).to(self.device)
    #         outp_vec = Y[i].unsqueeze(0).to(self.device)
    #         pred_vec = self.model(inp_vec).squeeze(0)
    #         loss = self.loss_fn(outp_vec, pred_vec)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         if i % 1000 == 0:
    #             print(loss.item(), '{}/{}'.format(i, len(X)))
    #         self.optimizer.step()
    #
    # def train_margin_epoch(self):
    #     """
    #     Train with max margin loss using the scrambled set
    #     :return:
    #     """
    #     self.model.train()
    #     X, Y, Y_hat = self.data.prepare_data(self.data.train_indices)
    #     for i in range(len(X)):
    #         inp_vec = batch(X[i]).to(self.device)
    #         outp_vec = Y[i].unsqueeze(0).to(self.device)
    #         pred_vec = self.model(inp_vec).squeeze(0)
    #         sc_vec = Y_hat[i].unsqueeze(0).to(self.device)
    #         t_loss = self.loss_fn(outp_vec, pred_vec)
    #         f_loss = self.loss_fn(sc_vec, pred_vec)
    #         loss = t_loss - f_loss + self.args.margin
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         if i % 1000 == 0:
    #             print(loss.item(), '{}/{}'.format(i, len(X)))
    #         self.optimizer.step()

    # def evaluate_likelihood_epoch(self, epoch=0):
    #     """
    #     Evaluate the trained metric
    #     :return:
    #     """
    #     self.model.eval()
    #     # calculate utterance scores on Test set
    #     X, Y, Y_hat = self.data.prepare_data(self.data.test_indices)
    #     test_scores = []
    #     test_scores_s = []
    #     for i in range(len(X)):
    #         inp_vec = batch(X[i]).to(self.device)
    #         outp_vec = Y[i].unsqueeze(0).to(self.device)
    #         sc_vec = Y_hat[i].unsqueeze(0).to(self.device)
    #         pred_vec = self.model(inp_vec).squeeze(0)
    #         loss = self.loss_fn(outp_vec, pred_vec)
    #         test_scores.append(loss.item())
    #         loss = self.loss_fn(sc_vec, pred_vec)
    #         test_scores_s.append(loss.item())
    #     print("Test, Mean : {}, Std : {}".format(np.mean(test_scores), np.std(test_scores)))
    #     print("Test Scrambled, Mean : {}, Std : {}".format(np.mean(test_scores_s), np.std(test_scores_s)))
    #     print("Difference mean : {}".format(np.mean(test_scores_s) - np.mean(test_scores)))

    # def train(self):
    #     """
    #     Run training for n epochs
    #     :return:
    #     """
    #     if self.args.load_model:
    #         self.load_model()
    #     for epoch in range(self.start_epoch, self.args.epochs):
    #         self.cur_epoch = epoch
    #         print("Epoch {}".format(epoch))
    #         if self.args.train_mode == 'likelihood':
    #             self.train_likelihood_epoch()
    #             self.evaluate_likelihood_epoch()
    #         elif self.args.train_mode == 'margin':
    #             self.train_margin_epoch()
    #             self.evaluate_likelihood_epoch()
    #         elif self.args.train_mode == 'binary':
    #             self.train_binary(epoch=epoch)
    #             self.evaluate_binary(epoch=epoch)
    #         else:
    #             raise NotImplementedError("training mode unknown")
    #         self.save_model()


if __name__ == "__main__":
    args = get_args()
    logbook = LogBook(vars(args))
    logbook.write_metadata_logs(vars(args))
    print("Loading {} data".format(args.data_name))
    data = ParlAIExtractor(args, logbook)
    data.load()
    dm = DialogMetric(args, data, logbook)
    dm.run()
