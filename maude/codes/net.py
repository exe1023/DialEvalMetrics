"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# Base Model with Uniform metrics and dataloaders
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import get_optimizer
from collections import OrderedDict
from transformers import BertModel, DistilBertModel
from data import ParlAIExtractor, DialogDataLoader, id_collate_fn, id_context_collate_fn
import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
import copy
from logbook.logbook import LogBook


class Net(LightningModule):
    def __init__(self, hparams, logbook=None):
        super(Net, self).__init__()
        self.hparams = hparams
        self.copy_hparams = copy.deepcopy(self.hparams)
        self.minibatch_step = 0
        self.bert_input = False  # whether the model needs bert style input
        self.is_transition_fn = False  # whether the model employs transition fn
        self.collate_fn = id_collate_fn
        self.context_collate_fn = id_context_collate_fn
        if logbook:
            self.logbook = logbook
        else:
            self.logbook = LogBook(vars(hparams))

    def forward(self, *args, **kwargs):
        pass

    def calc_ref_scores(self, batch):
        pass

    def calc_cont_scores(self, batch):
        # inp, inp_len, y_true, y_false, inp_false, inp_false_len = batch
        # pred_true = self.forward(inp, inp_len, y_true, mode='cont_score')
        # pred_false = self.forward(inp_false, inp_false_len, y_false, mode='cont_score')
        # return pred_true, pred_false
        pass

    def calc_nce_scores(self, batch):
        pass

    def calc_nce_scores_batched(self, batch):
        pass

    def calc_nce_scores_with_context_batched(self, batch):
        pass

    def init_bert_model(self):
        """
        Initialize bert pretrained or finetuned model
        :return:
        """
        # device = self.args.device
        load_path = self.hparams.bert_model
        if self.hparams.load_fine_tuned:
            load_path = self.hparams.fine_tune_model
        self.logbook.write_message_logs("Loading bert from {}".format(load_path))
        if self.hparams.bert_model == "bert-base-uncased":
            bert = BertModel.from_pretrained(load_path)
        elif self.hparams.bert_model == "distilbert-base-uncased":
            bert = DistilBertModel.from_pretrained(load_path)
        else:
            raise NotImplementedError("bert-model not implemented")
        # bert.to(device)
        return bert

    def extract_sentence_bert(self, sents, sent_len):
        """
        Extract sentence bert representation
        where sents is a batch of sentences
        :param sent:
        :return:
        """
        with torch.no_grad():
            if sents.dim() > 2:
                sents = sents.view(-1, sents.size(-1))
            batch, num_sent = sents.shape
            device = sents.device
            if sent_len.dim() > 1:
                sent_len = sent_len.view(-1)
            # segments_tensor = torch.zeros_like(sents).long().to(device)
            # batch indexes
            tokens_tensor = sents.detach()
            outs = self.bert(tokens_tensor)
            # outs = outs[1]
            outs = outs[0]
            # change: don't do mean pooling. according to the paper,
            # the CLS token representation has the sentence info
            return outs[:, 0]
            # sent_len = sent_len.float().unsqueeze(1).to(outs.device)
            # emb = torch.sum(outs, 1).squeeze(1)
            # emb = emb / sent_len.expand_as(emb)
            # # correct inf and -inf
            # emb[emb == float("Inf")] = 0
            # emb[emb == float("-Inf")] = 0
            # return emb.detach()

    def training_step(self, batch, batch_nb):
        # REQUIRED
        if self.hparams.train_mode == "nce":
            batched = True
            if batched:
                if self.hparams.corrupt_type == "all_context":
                    (
                        pred_true,
                        pred_false_scores,
                        target,
                    ) = self.calc_nce_scores_with_context_batched(batch)
                else:
                    pred_true, pred_false_scores, target = self.calc_nce_scores_batched(
                        batch
                    )
                pred_scores = torch.cat([pred_true, pred_false_scores], dim=0)
            else:
                pred_true, pred_false_scores = self.calc_nce_scores(batch)
                pred_scores = torch.cat([pred_true] + pred_false_scores, dim=0)
                true_weight = torch.ones_like(pred_true)
                true_weight = true_weight * len(pred_false_scores)
                target = torch.cat(
                    [torch.ones_like(pred_true)]
                    + [torch.zeros_like(pf) for pf in pred_false_scores],
                    dim=0,
                )
            loss = F.binary_cross_entropy(pred_scores, target, reduction="mean")
        else:
            if self.hparams.train_mode == "ref_score":
                pred_true, pred_false = self.calc_ref_scores(batch)
            else:
                pred_true, pred_false = self.calc_cont_scores(batch)
            device = pred_true.device
            t_loss = self.loss_fn(
                pred_true, torch.ones(pred_true.size(0), 1).to(device)
            )
            f_loss = self.loss_fn(
                pred_false, torch.zeros(pred_false.size(0), 1).to(device)
            )
            loss = t_loss + f_loss
        torch.cuda.empty_cache()
        # if batch_nb % 500 == 0:
        #     metrics = {
        #         "mode": "train",
        #         "minibatch": self.minibatch_step,
        #         "loss": loss.mean().item(),
        #         "true_loss": t_loss.mean().item(),
        #         "false_loss": f_loss.mean().item(),
        #         "epoch": self.trainer.current_epoch,
        #
        #     }
        #     self.minibatch_step += 1
        # self.logbook.write_metric_logs(metrics)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
        tqdm_dict = {"train_loss": loss}
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        if self.hparams.train_mode == "nce":
            batched = True
            if batched:
                if self.hparams.corrupt_type == "all_context":
                    (
                        pred_true,
                        pred_false_scores,
                        target,
                    ) = self.calc_nce_scores_with_context_batched(batch)
                else:
                    pred_true, pred_false_scores, target = self.calc_nce_scores_batched(
                        batch
                    )
                pred_false = pred_false_scores.mean(dim=-1)
                pred_scores = torch.cat([pred_true, pred_false_scores], dim=0)
            else:
                pred_true, pred_false_scores = self.calc_nce_scores(batch)
                pred_scores = torch.cat([pred_true] + pred_false_scores, dim=0)
                true_weight = torch.ones_like(pred_true)
                true_weight = true_weight * len(pred_false_scores)
                target = torch.cat(
                    [torch.ones_like(pred_true)]
                    + [torch.zeros_like(pf) for pf in pred_false_scores],
                    dim=0,
                )
                pred_false = torch.cat(pred_false_scores, dim=-1).mean(dim=-1)
            loss = F.binary_cross_entropy(pred_scores, target, reduction="mean")
        else:
            if self.hparams.train_mode == "ref_score":
                pred_true, pred_false = self.calc_ref_scores(batch)
            else:
                pred_true, pred_false = self.calc_cont_scores(batch)
            device = pred_true.device
            t_loss = self.loss_fn(
                pred_true, torch.ones(pred_true.size(0), 1).to(device)
            )
            f_loss = self.loss_fn(
                pred_false, torch.zeros(pred_false.size(0), 1).to(device)
            )
            loss = t_loss + f_loss
        torch.cuda.empty_cache()
        return OrderedDict(
            {"val_loss": loss, "true_score": pred_true, "false_score": pred_false}
        )

    def validation_end(self, outputs):
        # OPTIONAL
        # import ipdb; ipdb.set_trace()

        mean_dict = {"val_loss": 0, "true_score": 0, "false_score": 0}
        for output in outputs:
            for key in mean_dict:
                if key in ["minibatch", "epoch"]:
                    continue
                tmp = output[key]
                # if self.trainer.use_dp:
                tmp = torch.mean(tmp)
                mean_dict[key] += tmp

        for key in mean_dict:
            mean_dict[key] = mean_dict[key] / len(outputs)
        metrics = {
            "minibatch": self.minibatch_step,
            "epoch": self.trainer.current_epoch,
            "val_loss": mean_dict["val_loss"],
            "true_score": mean_dict["true_score"],
            "false_score": mean_dict["false_score"],
            "mode": 1,
        }
        self.minibatch_step += 1
        # self.logbook.write_metric_logs(metrics)
        result = {
            "progress_bar": metrics,
            "log": metrics,
            "val_loss": mean_dict["val_loss"],
        }
        return result

    def test_step(self, batch, batch_nb):
        # OPTIO
        with torch.no_grad():
            if self.hparams.train_mode == "nce":
                batched = True
                if batched:
                    if self.hparams.corrupt_type == "all_context":
                        (
                            pred_true,
                            pred_false_scores,
                            target,
                        ) = self.calc_nce_scores_with_context_batched(batch)
                    else:
                        (
                            pred_true,
                            pred_false_scores,
                            target,
                        ) = self.calc_nce_scores_batched(batch)
                    pred_false = pred_false_scores.mean(dim=-1)
                    pred_scores = torch.cat([pred_true, pred_false_scores], dim=0)
                else:
                    pred_true, pred_false_scores = self.calc_nce_scores(batch)
                    pred_scores = torch.cat([pred_true] + pred_false_scores, dim=0)
                    true_weight = torch.ones_like(pred_true)
                    true_weight = true_weight * len(pred_false_scores)
                    target = torch.cat(
                        [torch.ones_like(pred_true)]
                        + [torch.zeros_like(pf) for pf in pred_false_scores],
                        dim=0,
                    )
                    pred_false = torch.cat(pred_false_scores, dim=-1).mean(dim=-1)
                loss = F.binary_cross_entropy(pred_scores, target, reduction="mean")
            else:
                if self.hparams.train_mode == "ref_score":
                    pred_true, pred_false = self.calc_ref_scores(batch)
                else:
                    pred_true, pred_false = self.calc_cont_scores(batch)
                device = pred_true.device
                t_loss = self.loss_fn(
                    pred_true, torch.ones(pred_true.size(0), 1).to(device)
                )
                f_loss = self.loss_fn(
                    pred_false, torch.zeros(pred_false.size(0), 1).to(device)
                )
                loss = t_loss + f_loss
            torch.cuda.empty_cache()
            return OrderedDict(
                {"val_loss": loss, "true_score": pred_true, "false_score": pred_false}
            )

    def test_end(self, outputs):
        # OPTIONAL
        # import ipdb; ipdb.set_trace()

        mean_dict = {"val_loss": 0, "true_score": 0, "false_score": 0}
        for output in outputs:
            for key in mean_dict:
                if key in ["minibatch", "epoch"]:
                    continue
                tmp = output[key]
                # if self.trainer.use_dp:
                tmp = torch.mean(tmp)
                mean_dict[key] += tmp

        for key in mean_dict:
            mean_dict[key] = mean_dict[key] / len(outputs)
        metrics = {
            "minibatch": self.minibatch_step,
            "epoch": self.trainer.current_epoch,
            "val_loss": mean_dict["val_loss"],
            "true_score": mean_dict["true_score"],
            "false_score": mean_dict["false_score"],
            "mode": 2,
        }
        self.minibatch_step += 1
        # self.logbook.write_metric_logs(metrics)
        result = {
            "progress_bar": OrderedDict(metrics),
            "log": OrderedDict(metrics),
            "test_loss": metrics["val_loss"],
        }
        return result

    def preflight_steps(self):
        pass

    def get_dataloader(self, mode="train", datamode="train"):
        try:
            if datamode == "test" and mode == "train":
                raise AssertionError("datamode test does not have training indices")
            hparams = copy.deepcopy(self.copy_hparams)
            hparams.mode = datamode
            data = None
            # import ipdb; ipdb.set_trace()
            if datamode == "train":
                if not self.train_data:
                    self.logbook.write_message_logs("init loading data for training")
                    self.train_data = ParlAIExtractor(hparams, self.logbook)
                    self.hparams = self.train_data.args
                    self.train_data.load()
                    self.preflight_steps()
                    # if self.hparams.downsample:
                    #     self.logbook.write_message_logs("Downsampling to {}".format(
                    #         self.hparams.down_dim))
                    #     self.downsample()
                    #     self.data.clear_emb()
                data = self.train_data
            elif datamode == "test":
                self.logbook.write_message_logs("init loading data for testing")
                hparams.mode = "test"
                self.test_data = ParlAIExtractor(hparams, self.logbook)
                self.hparams = self.test_data.args
                self.test_data.args.load_model_responses = False
                self.test_data.load()
                self.preflight_steps()
                data = self.test_data
            if mode == "train":
                indices = data.train_indices
            elif mode == "test":
                indices = data.test_indices
            else:
                raise NotImplementedError("get_dataloader mode not implemented")
            ddl = DialogDataLoader(
                self.hparams,
                data,
                indices=indices,
                bert_input=self.bert_input,
                is_transition_fn=self.is_transition_fn,
            )
            ## ddl = DialogDiskDataLoader(self.hparams, mode, epoch)
            dist_sampler = None
            batch_size = self.hparams.batch_size
            # try:
            # if self.on_gpu:
            if self.use_ddp:
                dist_sampler = DistributedSampler(ddl, rank=self.trainer.proc_rank)
                batch_size = self.hparams.batch_size // self.trainer.world_size
            print(batch_size)
            # except Exception as e:
            #     pass
            if self.hparams.train_mode in ["ref_score", "nce"]:
                return DataLoader(
                    ddl,
                    collate_fn=self.collate_fn,
                    batch_size=batch_size,
                    sampler=dist_sampler,
                    num_workers=self.hparams.num_workers,
                )
            else:
                return DataLoader(
                    ddl,
                    collate_fn=self.context_collate_fn,
                    batch_size=batch_size,
                    sampler=dist_sampler,
                    num_workers=self.hparams.num_workers,
                )
        except Exception as e:
            print(e)
            # import ipdb; ipdb.set_trace()

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optim_fn, optim_params = get_optimizer(self.hparams.optim)
        optim = optim_fn(self.parameters(), **optim_params)
        # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer=optim,
        #         mode="min",
        #         patience=5,
        #         factor=0.8,
        #         verbose=True)
        # return torch.optim.Adam(self.parameters(), lr=0.02)
        # return [optim], [sched]
        return optim

    def set_hparam(self, key, val):
        setattr(self.hparams, key, val)
        setattr(self.copy_hparams, key, val)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        self.logbook.write_message_logs("fetching train dataloader ...")
        dl = self.get_dataloader(mode="train")
        print(isinstance(dl, DistributedSampler))
        return dl

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        self.logbook.write_message_logs("fetching valid dataloader ...")
        dl = self.get_dataloader(mode="test")
        print(isinstance(dl, DistributedSampler))
        return dl

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        self.logbook.write_message_logs("fetching test dataloader ...")
        return self.get_dataloader(mode="test", datamode="test")

    def tst_data(self, ddl, data):
        for di in ddl:
            assert max(di[0]) < len(data.tokenizer.vocab)
            assert max(di[1]) < len(data.tokenizer.vocab)
            assert max(di[2]) < len(data.tokenizer.vocab)
