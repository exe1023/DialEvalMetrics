"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
## Call various baselines from here
import torch
from backup.baselines.ruber.model import ReferencedMetric, UnreferencedMetric
from utils import batchify, BaseTrainer
import numpy as np
from args import get_args
from logbook.logbook import LogBook
from data import ParlAIExtractor
from scipy.stats.mstats import gmean


class RUBER(BaseTrainer):
    """Class combining referenced and unreference metrics
        There is no training required for Referenced metric. While validation,
        return the combination of Ref and Unref metrics
    """

    def __init__(self, args, data, logbook: LogBook):
        super().__init__(args, data, logbook)
        model = UnreferencedMetric(args, data, logbook)
        self.setup_model_opt(model)
        self.ref_model = ReferencedMetric(args, data, logbook)

    def train_epoch(self, epoch=0):
        """
        Train the unreferenced metric
        :return:
        """
        self.model.train()
        X, Y, Y_hat = self.data.prepare_data(self.data.train_indices, False)
        loss_a = []
        t_loss_a = []
        f_loss_a = []
        num_batches = len(list(range(0, len(X), self.args.batch_size)))
        for i in range(0, len(X), self.args.batch_size):
            inp_vec, inp_len = batchify(X[i : i + self.args.batch_size], False)
            outp_vec, outp_len = batchify(Y[i : i + self.args.batch_size], False)
            inp_vec = inp_vec.to(self.device)
            outp_vec = outp_vec.to(self.device)
            diff_true = self.model(inp_vec, inp_len, outp_vec, outp_len)
            y_false, y_len = batchify(Y_hat[i : i + self.args.batch_size], False)
            y_false = y_false.to(self.device)
            diff_false = self.model(inp_vec, inp_len, y_false, y_len)
            # import pdb; pdb.set_trace()
            loss = torch.clamp(
                torch.ones_like(diff_true) * self.args.margin - diff_true + diff_false,
                min=0.0,
            )
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            loss_a.append(loss.item())
            t_loss_a.append(diff_true.mean().item())
            f_loss_a.append(diff_false.mean().item())
            if i % self.args.log_interval == 0 or (i + 1) > (
                len(X) - self.args.batch_size
            ):
                print(i)
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

    def evaluate_epoch(self, epoch=0, mode="valid"):
        """
        Test with both unreferenced and referenced metric
        :return:
        """
        self.model.eval()
        X, Y, Y_hat = self.data.prepare_data(self.data.test_indices, False)
        loss_a = []
        t_loss_a = []
        f_loss_a = []
        ref_scores = []
        num_batches = len(list(range(0, len(X), self.args.batch_size)))
        for i in range(0, len(X), self.args.batch_size):
            inp_vec, inp_len = batchify(X[i : i + self.args.batch_size], False)
            outp_vec, outp_len = batchify(Y[i : i + self.args.batch_size], False)
            inp_vec = inp_vec.to(self.device)
            outp_vec = outp_vec.to(self.device)
            diff_true = self.model(inp_vec, inp_len, outp_vec, outp_len)
            y_false, y_len = batchify(Y_hat[i : i + self.args.batch_size], False)
            y_false = y_false.to(self.device)
            diff_false = self.model(inp_vec, inp_len, y_false, y_len)
            loss = torch.clamp(
                torch.ones_like(diff_true) * self.args.margin - diff_true + diff_false,
                min=0.0,
            )
            loss = loss.mean()
            loss_a.append(loss.item())
            t_loss_a.append(diff_true.mean().item())
            f_loss_a.append(diff_false.mean().item())
            ref_score = self.ref_model.score(outp_vec, outp_len, y_false, y_len)
            ref_scores.append(ref_score.mean().item())

        metrics = {
            "mode": mode,
            "minibatch": self.train_step,
            "loss": np.mean(loss_a),
            "true_loss": np.mean(t_loss_a),
            "false_loss": np.mean(f_loss_a),
            "ref_scores": np.mean(ref_scores),
            "epoch": epoch,
        }
        self.logbook.write_metric_logs(metrics)

    def calc_ruber(self, diff_false, ref_score):
        s_r = torch.norm(diff_false)
        s_u = torch.norm(ref_score)
        min_r = torch.min(s_r, s_u)[0].numpy()
        max_r = torch.max(s_r, s_u)[0].numpy()
        am_r = torch.mean(s_r, s_u).numpy()
        gp_r = gmean(torch.cat([s_r, s_u], axis=1).numpy(), axis=1)
        return min_r, max_r, am_r, gp_r


if __name__ == "__main__":
    args = get_args()
    logbook = LogBook(vars(args))
    logbook.write_metadata_logs(vars(args))
    print("Loading {} data".format(args.data_name))
    data = ParlAIExtractor(args, logbook)
    data.load()
    data.load_tokens()
    ruber = RUBER(args, data, logbook)
    ruber.run()

