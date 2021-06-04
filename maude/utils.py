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
import random
import numpy as np
import re
import inspect
from torch import optim
import importlib
import os
from transformers import BertModel


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def getbuckets(Y, n_buckets):
    """
    Get n_buckets of conversations
    :param Y:
    :param n_buckets:
    :return:
    """
    Y_new = []
    # min-max list for buckets
    incr = int(100 / n_buckets)
    buckets = [(i + 1, i + incr) for i in range(0, 100, incr)]
    for y in Y:
        for i in range(len(buckets)):
            if y >= buckets[i][0] and y <= buckets[i][1]:
                Y_new += [i]
    return np.stack(Y_new, axis=0)


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    Source: InferSent
    """
    if "," in s:
        method = s[: s.find(",")]
        optim_params = {}
        for x in s[s.find(",") + 1 :].split(","):
            split = x.split("=")
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == "adadelta":
        optim_fn = optim.Adadelta
    elif method == "adagrad":
        optim_fn = optim.Adagrad
    elif method == "adam":
        optim_fn = optim.Adam
    elif method == "adamax":
        optim_fn = optim.Adamax
    elif method == "asgd":
        optim_fn = optim.ASGD
    elif method == "rmsprop":
        optim_fn = optim.RMSprop
    elif method == "rprop":
        optim_fn = optim.Rprop
    elif method == "sgd":
        optim_fn = optim.SGD
        assert "lr" in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ["self", "params"]
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception(
            'Unexpected parameters: expected "%s", got "%s"'
            % (str(expected_args[2:]), str(optim_params.keys()))
        )

    return optim_fn, optim_params


def batch(vecs):
    return torch.stack(vecs).unsqueeze(0)


def batchify(data, vector_mode=True):
    if vector_mode:
        return batch_vectors(data)
    else:
        return batch_words(data)


def batch_dialogs(dials):
    """
    Batch dialogs
    :param dials:
    :return:
    """
    # print(type(dials))
    # assert type(dials) == list
    # assert type(dials[0]) == list
    dial_length = [len(s) for s in dials]
    word_length = [[len(sent) for sent in dial] for dial in dials]
    wmax = max([c for p in word_length for c in p])
    mat = torch.zeros((len(dials), max(dial_length), wmax)).long()
    wlen = torch.zeros((len(dials), max(dial_length)))
    for i, dial in enumerate(dials):
        dial_end = dial_length[i]
        for j, sent in enumerate(dial):
            sent_end = word_length[i][j]
            mat[i, j, :sent_end] = torch.LongTensor(sent[:sent_end])
            wlen[i][j] = sent_end
    return mat, np.array(dial_length), wlen


def batch_words(dials):
    """
    for each dialog, join all words into one big sentence
    :param words:
    :return:
    """
    if type(dials[0]) != list:
        dials = [dials]
    fl_length = [len(s) for s in dials]
    mat = torch.zeros(len(dials), max(fl_length)).long()
    for i, sents in enumerate(dials):
        end = fl_length[i]
        mat[i, :end] = torch.LongTensor(sents[:end])
    return mat, np.array(fl_length)


def batch_vectors(vecs):
    """
    batchify vectors and return the dialog length
    :param vecs:
    :return:
    """
    if type(vecs[0]) == list:
        dim = vecs[0][0].size(0)
        lengths = [len(v) for v in vecs]
        mat = torch.zeros(len(lengths), max(lengths), dim)
        for i, vec in enumerate(vecs):
            for j, v in enumerate(vec):
                mat[i, j, :] = v
    else:
        lengths = [1 for v in vecs]
        mat = torch.stack(vecs, dim=0).unsqueeze(1)
    return mat, lengths


def batch_yhats(yhats):
    """
    Expects yhats to be 
    [
        [[],[],[],[]], --> k different negative yhats
    ]
    returns mat = B x k x max_words
            length = B x k x 1
    """
    assert type(yhats[0]) == list
    assert type(yhats[0][0]) == list
    B = len(yhats)
    k = len(yhats[0])
    lengths = [len(sent) for row in yhats for sent in row]
    mat = torch.zeros(B, k, max(lengths)).long()
    lmat = torch.zeros(B, k, 1).long()
    for ri, rows in enumerate(yhats):
        for si, sent in enumerate(rows):
            for wi, word in enumerate(sent):
                mat[ri, si, wi] = word
            lmat[ri, si, 0] = len(sent)
    return mat, lmat


def _import_module(full_module_name):
    """
    Import className from python file
    https://stackoverflow.com/a/8790232
    :param full_module_name: full resolvable module name
    :return: module
    """
    path, name = full_module_name.rsplit(".", 1)
    base_module = importlib.import_module(path)
    module = getattr(base_module, name)
    return module


class BaseTrainer:
    """
    Base trainer class to be used by all trainers
    """

    def __init__(self, args, data, logbook):
        self.args = args
        self.data = data
        self.logbook = logbook
        self.device = torch.device(args.device)
        self.optimizer = None
        self.train_step = 0
        self.valid_step = 0
        self.cur_epoch = 0
        self.start_epoch = 0
        self.model = None

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

    def setup_model_opt(self, model):
        """
        Setup model and optimizers
        :param model:
        :return:
        """
        self.model = model
        self.model.to(self.device)
        # self.model = nn.DataParallel(self.model, device_ids=[0,1])
        optim_fn, optim_params = get_optimizer(self.args.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)

    def save_model(self, is_best_model=False, index=0):
        """
        Save model and information
        :param is_best_model:
        :param index:
        :return:
        """
        state = {
            "train_step": self.train_step + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "np_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "pytorch_random_state": torch.get_rng_state(),
            "index": index,
            "epoch": self.cur_epoch,
            "model_name": self.args.model,
            "word_dict": self.data.word_dict,
            "train_indices": self.data.train_indices,
            "test_indices": self.data.test_indices
            # "schedulers": [scheduler.state_dict() for scheduler in schedulers]
        }
        if is_best_model:
            path = os.path.join(
                self.args.model_save_dir,
                "best",
                "{}_agent_id_{}.tar".format(self.args.id, index),
            )
        else:
            path = os.path.join(
                self.args.model_save_dir,
                "{}_agent_id_{}.tar".format(self.args.id, index),
            )

        torch.save(state, path)
        self.logbook.write_message_logs("saved model to path = {}".format(path))

    def load_model(self, index=0, should_load_optimizer=False):
        """
        Load model and information
        :param index:
        :param should_load_optimizer:
        :return:
        """
        load_path = self.args.model_load_path
        if load_path[-1] == "/":
            load_path = load_path[:-1]
        path = "{}/{}_agent_id_{}.tar".format(
            self.args.model_save_dir, self.args.id, index
        )
        if not os.path.exists(path):
            path = "{}/{}_agent_id_{}.tar".format(load_path, self.args.id, 0)
        self.logbook.write_message_logs("Loading model from path {}".format(path))
        if str(self.args.device) == "cuda":
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        # assert if we are loading the same model
        assert self.args.model == checkpoint["model_name"]
        self.cur_epoch = checkpoint["epoch"]
        self.train_step = checkpoint["train_step"]
        np.random.set_state(checkpoint["np_random_state"])
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["pytorch_random_state"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.data.word_dict = checkpoint["word_dict"]
        import ipdb

        ipdb.set_trace()
        if self.args.mode == "train":
            self.data.train_indices = checkpoint["train_indices"]
            self.data.test_indices = checkpoint["test_indices"]

        if should_load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.logbook.write_message_logs("Done loading model from path {}".format(path))

    def train_epoch(self, epoch=0):
        """
        to be implemented
        :return:
        """
        pass

    def evaluate_epoch(self, epoch=0, mode="valid"):
        """
        to be implemented
        :param epoch:
        :return:
        """
        pass

    def run(self):
        """
        Main running method
        :return:
        """
        if self.args.eval_val:
            self.load_model()
            self.evaluate_epoch()
            return
        if self.args.load_model or self.args.mode == "test":
            self.load_model()
        if self.args.mode == "train":
            for epoch in range(self.start_epoch, self.args.epochs):
                self.cur_epoch = epoch
                print("Epoch {}".format(epoch))
                self.train_epoch()
                self.evaluate_epoch()
                self.save_model()
        elif self.args.mode == "test":
            import ipdb

            ipdb.set_trace()
            self.evaluate_epoch(mode="test")
        else:
            raise NotImplementedError(
                "args.mode {} not implemented".format(self.args.mode)
            )

