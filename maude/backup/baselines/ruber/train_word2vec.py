"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# Code to train word2vec models from scratch, as required in the paper, for Referenced Metric
# Optimized from Srijith Rajamohan, https://srijithr.gitlab.io/post/word2vec/
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from nltk.corpus import stopwords
from args import get_args
from logbook.logbook import LogBook
from data import ParlAIExtractor


class CBOWModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOWModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(
            inputs.size(0), -1
        )  # -1 implies size inferred for that index from the size of the data
        # print(np.mean(np.mean(self.linear2.weight.data.numpy())))
        out1 = F.relu(self.linear1(embeds))  # output of first layer
        out2 = self.linear2(out1)  # output of second layer
        # print(embeds)
        log_probs = F.log_softmax(out2, dim=1)
        return log_probs


class Word2VecTrainer:
    """Train word2vec models"""

    def __init__(self, args, data, log: LogBook):
        self.args = args
        self.data = data
        self.log = log
        self.losses = []
        self.loss_function = nn.NLLLoss()
        self.model = CBOWModeler(
            len(data.word_dict), args.word2vec_embedding_dim, args.word2vec_context_size
        )
        self.device = torch.device(self.args.device)
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.word2vec_lr)
        self.prepare_dataset()
        self.train_step = 0

    def prepare_dataset(self):
        """
        For word2vec training, concatenate all text into one giant blob
        Then, create n-grams out of the blob
        :return:
        """
        self.log.write_message_logs("Re-formatting dataset ...")
        all_tokens = [
            word for dial in self.data.dialog_tokens for utt in dial for word in utt
        ]
        stop_words = set(stopwords.words("english"))
        stop_words.update([".", ",", ":", ";", "(", ")", "#", "--", "...", '"'])
        cleaned_words = [i for i in all_tokens if i not in stop_words]
        ngrams = []
        for i in range(len(all_tokens) - self.args.word2vec_context_size):
            tup = [
                all_tokens[j] for j in np.arange(i, i + self.args.word2vec_context_size)
            ]
            ngrams.append((tup, all_tokens[i + self.args.word2vec_context_size]))
        self.ngrams = ngrams
        self.log.write_message_logs("{} ngrams computed".format(len(ngrams)))

    def train(self):
        self.log.write_message_logs("starting training ...")
        for epoch in range(self.args.word2vec_epochs):
            losses = []
            # ------- Embedding layers are trained as well here ----#
            # lookup_tensor = torch.tensor([word_to_ix["poor"]], dtype=torch.long)
            # hello_embed = model.embeddings(lookup_tensor)
            # print(hello_embed)
            # -----------------------------------------------------#
            num_batches = len(range(0, len(self.ngrams), self.args.word2vec_batchsize))
            self.log.write_message_logs("Number of batches : {}".format(num_batches))
            for minibatch in range(0, len(self.ngrams), self.args.word2vec_batchsize):
                contexts, targets = zip(
                    *self.ngrams[minibatch : minibatch + self.args.word2vec_batchsize]
                )
                context_idxs = torch.tensor(
                    [
                        [self.data.get_word_id(w) for w in context]
                        for context in contexts
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                self.optimizer.zero_grad()
                log_probs = self.model(context_idxs)
                y = torch.tensor(
                    [[self.data.get_word_id(target)] for target in targets],
                    dtype=torch.long,
                    device=self.device,
                )
                y = y.squeeze(1)
                loss = self.loss_function(log_probs, y)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.mean().item())

                if minibatch % 1000 == 0:
                    metrics = {
                        "mode": "train",
                        "loss": np.mean(losses),
                        "epoch": epoch,
                        "minibatch": self.train_step,
                    }
                    self.train_step += 1
                    self.log.write_metric_logs(metrics)
                    losses = []

            # done

            metrics = {
                "mode": "train",
                "loss": np.mean(losses),
                "epoch": epoch,
                "minibatch": self.train_step,
            }
            self.train_step += 1
            self.log.write_metric_logs(metrics)
            self.losses.append(np.mean(losses))
            self.save_embedding()

    def predict(self, input):
        context_idxs = torch.tensor(
            [self.data.get_word_id(w) for w in input], dtype=torch.long
        )
        res = self.model(context_idxs)
        res_arg = torch.argmax(res)
        res_val, res_ind = res.sort(descending=True)
        res_val = res_val[0][:3]
        res_ind = res_ind[0][:3]
        # print(res_val)
        # print(res_ind)
        for arg in zip(res_val, res_ind):
            # print(arg)
            print(
                [
                    (key, val, arg[0])
                    for key, val in self.data.word_dict.items()
                    if val == arg[1]
                ]
            )

    def freeze_layer(self, layer):
        for name, child in self.model.named_children():
            print(name, child)
            if name == layer:
                for names, params in child.named_parameters():
                    print(names, params)
                    print(params.size())
                    params.requires_grad = False

    def print_layer_parameters(self):
        for name, child in self.model.named_children():
            print(name, child)
            for names, params in child.named_parameters():
                print(names, params)
                print(params.size())

    def save_embedding(self):
        torch.save(self.model.embeddings.weight, self.args.word2vec_out)


if __name__ == "__main__":
    args = get_args()
    logbook = LogBook(vars(args))
    logbook.write_metadata_logs(vars(args))
    print("Loading {} data".format(args.data_name))
    data = ParlAIExtractor(args, logbook)
    data.load()
    data.load_tokens()
    trainer = Word2VecTrainer(args, data, logbook)
    trainer.train()

