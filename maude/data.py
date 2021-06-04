  
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pickle as pkl
import os
import random
from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.agents.ir_baseline.ir_baseline import IrBaselineAgent
from parlai.core.worlds import create_task
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
# from nltk.tokenize import sent_tokenize
import copy
from logbook.logbook import LogBook
from args import get_args
from parlai.core.agents import create_agent
from backup.corrupt import CorruptDialog
from utils import batchify, batch_dialogs, batch_words, batch_yhats
import hashlib
import pandas as pd

fixed_suffixes = ["true_response", "seq2seq","backtranslate"]
variable_suffixes = ["model_false", "rand_utt","rand_back","word_drop", "word_order","word_repeat", "corrupt_context"]
## positive sampling scheme
all_pos = ["true_response","backtranslate"]
## negative sampling schemes
only_syntax = ["word_drop","word_order","word_repeat"]
only_semantics = ["model_false","rand_utt","rand_back"]
all_corrupt = only_syntax + only_semantics
all_corrupt_context = only_syntax + only_semantics + ["corrupt_context"]


class Data:
    def __init__(self, args, logbook):
        args = self.set_file_paths(args)
        self.args = args
        self.logbook = logbook
        load_path = args.bert_model
        if args.load_fine_tuned:
            load_path = args.fine_tune_model
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.dialogs = {}
        self.dialog_tokens = []
        self.dial_vecs = []
        self.scramble_vecs = []
        self.train_indices = []
        self.test_indices = []
        self.down_model = ""
        # special word dict (not to be confused with bert token ids) for training baselines
        self.word_dict = {}
        self.interactions = []
        self.interaction_hashes = []
        self.hash2reponses = {}
        self.all_hashes = set()
        self.model_responses = {}
        # add special characters
        self.add_words(["[CLS]", "[SEP]", "UNK"])
        # if self.args.corrupt_type != "rand_utt":
        #     self.init_bert_model()

    def set_file_paths(self, args):
        """
        Set correct file paths
        :param args:
        :return:
        """
        if not args.pca_file.endswith(".pkl"):
            args.pca_file = args.data_loc + args.data_name + "_train_pca.pkl"
        if not args.model_response_pre.endswith(".pkl"):
            args.model_response_pre = args.data_loc + "{}_{}_store.pkl".format(
                args.data_name, args.mode
            )
        # if args.exp_data_folder == "na":
        #     args.exp_data_folder = os.path.join(args.data_loc, "{}_data".format(args.mode), args.id)
        #     if not os.path.exists(args.exp_data_folder):
        #         os.mkdir(args.exp_data_folder)
        finetuned = ""
        if args.load_fine_tuned:
            finetuned = "_finetuned_" + args.trained_bert_suffix
        if not args.emb_file.endswith(".pkl"):
            args.emb_file = (
                args.emb_file + args.data_name + finetuned + "_{}.pkl".format(args.mode)
            )
        # if not args.data_loc.endswith('.pkl'):
        #     args.data_loc = args.data_loc + args.data_name + \
        #                     '_{}.pkl'.format(args.mode)
        if not args.tok_file.endswith(".pkl"):
            args.tok_file = (
                args.data_loc + args.data_name + "_tokens_{}.pkl".format(args.mode)
            )
        return args

    def extract_tokens(self):
        """
        Extracting tokens
        :return:
        """
        self.logbook.write_message_logs("Extracting tokens")
        dialogs = self.dialogs
        tdialogs = [[self.tokenizer.tokenize(utt) for utt in dl] for dl in dialogs]
        self.dialog_tokens = tdialogs
        self.logbook.write_message_logs("Building vocab...")
        # add the words in our dictionary
        for dl in tdialogs:
            for utt in dl:
                self.add_words(utt)
        self.logbook.write_message_logs(
            "Extracted {} words".format(len(self.word_dict))
        )

    def extract_dialogs(self):
        """
        Should implement the method to load and extract raw dialogs for the dataset here
        :return:
        """
        pass

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
        self.bert.eval()
        # self.bert.to(self.args.device)

    def extract_sentence_bert(self, sents, tokenize=True):
        """
        Extract sentence bert representation
        where sents is a batch of sentences
        :param sent:
        :return:
        """
        if tokenize:
            tokens = [self.tokenizer.tokenize(sent) for sent in sents]
        else:
            tokens = sents
        indexes = [self.tokenizer.convert_tokens_to_ids(sent) for sent in tokens]
        max_indx_len = max([len(sent) for sent in tokens])
        segments_tensor = (
            torch.zeros(len(sents), max_indx_len).long().to(self.args.device)
        )
        # batch indexes
        tokens_tensor = (
            torch.zeros(len(sents), max_indx_len).long().to(self.args.device)
        )
        for i, indx in enumerate(indexes):
            tokens_tensor[i][: len(indx)] = torch.LongTensor(indx)
        with torch.no_grad():
            outs = self.bert(tokens_tensor, token_type_ids=segments_tensor)
            return outs[1].to("cpu")

    def extract_be(self):
        """
        Extract bert embeddings (common for all types of datasets)
        :return:
        """
        dialogs = self.dialogs
        self.logbook.write_message_logs("Tokenizing ...")
        tdialogs = [[self.tokenizer.tokenize(utt) for utt in dl] for dl in dialogs]
        index_dial = [
            [self.tokenizer.convert_tokens_to_ids(utt) for utt in dl] for dl in tdialogs
        ]
        segment_dial = [[[0 for tok in utt] for utt in dl] for dl in index_dial]
        # initalize bert model
        self.init_bert_model()
        self.logbook.write_message_logs("Extracting {} dialogs".format(len(dialogs)))
        pb = tqdm(total=len(dialogs))
        for di, dial in enumerate(index_dial):
            utt_vecs = []
            for uid, utt in enumerate(dial):
                tokens_tensor = torch.tensor([utt])
                segments_tensor = torch.tensor([segment_dial[di][uid]])
                tokens_tensor = tokens_tensor.to("cuda")
                segments_tensor = segments_tensor.to("cuda")
                with torch.no_grad():
                    outs = self.bert(tokens_tensor, token_type_ids=segments_tensor)
                utt_vecs.append(outs[1][0].to("cpu"))
            self.dial_vecs.append(utt_vecs)
            pb.update(1)
        pb.close()

    def add_words(self, words):
        assert type(words) == list
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        assert type(word) == str
        if word not in self.word_dict:
            self.word_dict[word] = len(self.word_dict)

    def get_word_id(self, word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return self.word_dict["UNK"]

    def save_dialog(self):
        """
        Save dialogs
        :return:
        """
        pkl.dump(
            {
                "raw": self.dialogs,
                "train_indices": self.train_indices,
                "test_indices": self.test_indices,
            },
            open(self.args.data_loc, "wb"),
        )

    def save_emb(self):
        """
        Save embeddings
        :return:
        """
        pkl.dump({"vec": self.dial_vecs}, open(self.args.emb_file, "wb"))

    def save_tokens(self):
        """
        Save word dict and tokens
        :return:
        """
        pkl.dump(
            {"word_dict": self.word_dict, "tokens": self.dialog_tokens},
            open(self.args.tok_file, "wb"),
        )

    def save_pca_model(self):
        """
        Save pca model
        :return:
        """
        pkl.dump({"pca": self.down_model}, open(self.args.pca_file, "wb"))

    def load_dialog(self):
        """
        Load dialogs
        Load all true and corrupt dialogs here
        :return:
        """

        # if os.path.exists(self.args.data_loc) and os.path.isfile(self.args.data_loc):
        #     self.logbook.write_message_logs("Loading dialogs from {}".format(self.args.data_loc))
        #     data_dump = pkl.load(open(self.args.data_loc,'rb'))
        #     self.dialogs = data_dump['raw']
        #     self.train_indices = data_dump['train_indices']
        #     self.test_indices = data_dump['test_indices']
        #     self.logbook.write_message_logs("Loaded {} dialogs".format(len(self.dialogs)))
        # else:
        #     self.logbook.write_message_logs("Extracting dialogs")
        #     self.extract_dialogs()
        #     self.logbook.write_message_logs("Extracted {} dialogs".format(len(self.dialogs)))
        #     self.split_train_test()
        #     self.save_dialog()
        for fs in fixed_suffixes:
            file_path = os.path.join(
                self.args.data_loc,
                "{}_{}_{}.csv".format(self.args.data_name, self.args.mode, fs),
            )
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.dialogs[fs] = pd.read_csv(file_path)
            else:
                raise FileNotFoundError("file {} not found".format(file_path))
        ep_id = 0
        found = True
        while found:
            for fs in variable_suffixes:
                new_fs = "{}_{}".format(fs, ep_id)
                file_path = os.path.join(
                    self.args.data_loc,
                    "{}_{}_{}.csv".format(self.args.data_name, self.args.mode, new_fs),
                )
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    self.dialogs[new_fs] = pd.read_csv(file_path)
                else:
                    found = False
                    break
            ep_id += 1
        self.check_nans()
        print("all dialog files loaded for mode {}".format(self.args.mode))

    def check_nans(self):
        """
        check for nans in the dataset
        :return:
        """
        null_found = False
        for dial_file, dial_csv in self.dialogs.items():
            if dial_csv.isnull().values.any():
                num_null = dial_csv.isnull().sum().sum()
                print("{} null items found in {}".format(num_null, dial_file))
                null_found = True
        if null_found:
            raise AssertionError("Check data and fix the null values")

    def load_emb(self):
        """
        Load bert embeddings, extract if not present
        :return:
        """
        if os.path.exists(self.args.emb_file) and os.path.isfile(self.args.emb_file):
            self.logbook.write_message_logs(
                "Loading embeddings from {}".format(self.args.emb_file)
            )
            data_dump = pkl.load(open(self.args.emb_file, "rb"))
            self.dial_vecs = data_dump["vec"]
            self.logbook.write_message_logs(
                "Loaded embeddings {}".format(len(self.dial_vecs))
            )
        else:
            print("Extracting BERT embeddings")
            self.extract_be()
            print("Saving embeddings")
            self.save_emb()

    def clear_emb(self):
        """
        Clear embeddings from memory
        :return:
        """
        del self.dial_vecs

    def load_tokens(self):
        """
        Load tokens
        :return:
        """
        if os.path.exists(self.args.tok_file) and os.path.isfile(self.args.tok_file):
            print("Loading tokens from {}".format(self.args.tok_file))
            data_dump = pkl.load(open(self.args.tok_file, "rb"))
            self.dialog_tokens = data_dump["tokens"]
            self.word_dict = data_dump["word_dict"]
        else:
            self.extract_tokens()
            self.save_tokens()

    def convert_tokens_to_ids(self, tokens):
        return [self.get_word_id(tok) for tok in tokens]

    def load_pca(self):
        """
        Load pca trained model
        :return:
        """
        if os.path.exists(self.args.pca_file) and os.path.isfile(self.args.pca_file):
            self.logbook.write_message_logs(
                "Loading pca model from {}".format(self.args.pca_file)
            )
            data_dump = pkl.load(open(self.args.pca_file, "rb"))
            self.down_model = data_dump["pca"]
        else:
            raise FileNotFoundError(
                "trained pca model doesn't exist. retrain with train split"
            )

    def load_model_responses(self):
        """
        load model responses
        :return:
        """
        if os.path.exists(self.args.model_response_pre) and os.path.isfile(
            self.args.model_response_pre
        ):
            self.logbook.write_message_logs(
                "Loading model responses from {}".format(self.args.model_response_pre)
            )
            self.model_responses = pkl.load(open(self.args.model_response_pre, "rb"))
        else:
            raise FileNotFoundError("model responses not exist")

    def load(self):
        """
        load dialogs and embeddings
        :return:
        """
        self.load_dialog()
        self.split_train_test()
        # self.load_emb()
        # if self.args.mode == 'test' or self.args.eval_val:
        #     self.load_pca()
        # if self.args.load_model_responses:
        #     self.load_model_responses()

    def get_dataloader(self, mode="train", epoch=0):
        """
        Get train/test dataloader
        :param mode:
        :return:
        """
        if mode == "train":
            indices = self.train_indices
        elif mode == "test":
            indices = self.test_indices
        else:
            raise NotImplementedError("get_dataloader mode not implemented")
        ddl = DialogDataLoader(self.args, self, indices=indices)
        ## ddl = DialogDiskDataLoader(self.args, mode, epoch)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(ddl)
        return DataLoader(
            ddl,
            collate_fn=id_collate_fn,
            num_workers=self.args.dataloader_threads,
            batch_size=self.args.batch_size,
            sampler=dist_sampler,
        )

    def prepare_data(self, mode, online=False):
        """
        Pre-process data and save them in disk
        :param mode: train or test
        :param vector: return the bert sentence vector of dialogs, else return raw words
        :param online: if vector, and if online, return vectors directly querying bert
        :return:
        """
        vector = self.args.vector_mode
        if mode == "train":
            indices = self.train_indices
        elif mode == "test":
            indices = self.test_indices
        else:
            raise NotImplementedError("{} not implemented".format(mode))
        if vector:
            if online:
                dialogs = [self.dialogs[di] for di in indices]
                dial_vecs = [self.extract_sentence_bert(dial) for dial in dialogs]
            else:
                dial_vecs = [self.dial_vecs[di] for di in indices]
        else:
            dial_vecs = [self.dialog_tokens[di] for di in indices]
            dial_vecs = [
                [[self.get_word_id(w) for w in utt] for utt in dl] for dl in dial_vecs
            ]
        dialogs = [self.dialogs[di] for di in indices]
        cd = CorruptDialog(self.args, self, False, bert_tokenize=True)
        # save individual epoch data in file
        pbe = tqdm(total=self.args.epochs)
        for epoch in range(self.args.epochs):
            X = []
            Y = []
            Y_hat = []
            pb = tqdm(total=len(dial_vecs))
            for di, dial in enumerate(dial_vecs):
                dialog_id = indices[di]
                for i in range(1, len(dial)):
                    inp = dial[0:i]
                    outp = dial[i]
                    if not vector:
                        # flatten into one sentence
                        inp = [w for utt in inp for w in utt]
                    X.append(inp)
                    Y.append([outp])
                    if self.args.corrupt_type == "rand_utt":
                        sc = cd.random_clean(dialog_id=dialog_id)
                    elif self.args.corrupt_type == "drop":
                        sc = cd.random_drop(
                            self.dialogs[dialog_id][i], drop=self.args.drop_per
                        )
                    elif self.args.corrupt_type == "shuffle":
                        sc = cd.change_word_order(self.dialogs[dialog_id][i])
                    elif self.args.corrupt_type in ["model_true", "model_false"]:
                        sc = cd.get_nce_semantics(dialog_id, i)
                    else:
                        raise NotImplementedError(
                            "args.corrupt_type {} not implemented".format(
                                self.args.corrupt_type
                            )
                        )
                    Y_hat.append(sc)
                pb.update(1)
            pb.close()
            # extract Y_hat from BERT
            Y_hat_h = []
            bs = 32
            self.logbook.write_message_logs("Extracting negative samples from BERT")
            pb = tqdm(total=len(range(0, len(Y_hat), bs)))
            for yi in range(0, len(Y_hat), bs):
                Y_hat_h.append(
                    self.pca_predict(
                        [
                            list(
                                self.extract_sentence_bert(
                                    Y_hat[yi : yi + bs], tokenize=False
                                )
                            )
                        ]
                    )[0]
                )
                pb.update(1)
            pb.close()
            epoch_data = [X, Y, Y_hat_h]
            pkl.dump(
                epoch_data,
                open(
                    os.path.join(
                        self.args.exp_data_folder, "{}_epoch_{}.pkl".format(mode, epoch)
                    ),
                    "wb",
                ),
            )
            pbe.update(1)
        pbe.close()

    def scramble(self, indices=None):
        """
        Scramble the last utterance of a dialog
        :return:
        """
        scrambled = []
        if indices:
            to_scramble = [d for di, d in enumerate(self.dial_vecs) if di in indices]
        else:
            to_scramble = copy.copy(self.dial_vecs)
        for i, dial in enumerate(to_scramble):
            sd = dial
            candidates = list(range(len(to_scramble)))
            candidates.remove(i)
            cand_dial = random.choice(candidates)
            sd[-1] = to_scramble[cand_dial][-1]
            scrambled.append(sd)
        return scrambled

    def split_train_test(self, ratio=0.9, force=False):
        """
        Split training and testing data in dialog level
        :return:
        """
        dialogs = self.dialogs["true_response"]
        if "split" not in dialogs or force:
            dialog_ids = list(dialogs["dialog_id"].unique())
            if self.args.mode == "train":
                tr_dv = random.sample(dialog_ids, int(len(dialog_ids) * ratio))
                ts_dv = [i for i in range(len(dialog_ids)) if i not in tr_dv]
                train_indices = [
                    i for i, row in dialogs.iterrows() if row["dialog_id"] in tr_dv
                ]
                test_indices = [
                    i for i, row in dialogs.iterrows() if row["dialog_id"] in ts_dv
                ]
                self.train_indices = train_indices
                self.test_indices = test_indices
            else:
                self.train_indices = []
                self.test_indices = list(range(len(dialogs)))
            self.logbook.write_message_logs(
                "Split done. Train rows : {}, Test rows : {}".format(
                    len(self.train_indices), len(self.test_indices)
                )
            )
            for i, row in dialogs.iterrows():
                split = "test"
                if i in self.train_indices:
                    split = "train"
                self.dialogs["true_response"].at[i, "split"] = split
            file_path = os.path.join(
                self.args.data_loc,
                "{}_{}_{}.csv".format(
                    self.args.data_name, self.args.mode, "true_response"
                ),
            )
            self.dialogs["true_response"].to_csv(file_path)
        else:
            # load the split from the data
            self.train_indices = []
            self.test_indices = []
            for i, row in self.dialogs["true_response"].iterrows():
                if row["split"] == "train":
                    self.train_indices.append(i)
                else:
                    self.test_indices.append(i)
            # indices = dialogs['split'].tolist()
            # self.train_indices = [i for i,r in enumerate(indices) if r == 'train']
            # self.test_indices = [i for i,r in enumerate(indices) if r == 'test']

    def pca_train(self):
        """
        Train pca on the training split at the beginning and store the
        model in memory / persist
        :return:
        """
        self.logbook.write_message_logs("Training PCA ..")
        dial_vecs = [self.dial_vecs[di] for di in self.train_indices]
        all_vecs = [d.numpy() for dial in dial_vecs for d in dial]
        all_vec_ids = [di for di, dial in enumerate(dial_vecs) for d in dial]
        tokens = np.array(all_vecs)
        self.down_model = PCA(n_components=self.args.down_dim, whiten=True)
        self.down_model.fit(tokens)

    def pca_predict(self, dial_vecs):
        """
        Predict given the vecs
        :param dial_vecs:
        :return:
        """
        all_vecs = [d.numpy() for dial in dial_vecs for d in dial]
        all_vec_ids = [di for di, dial in enumerate(dial_vecs) for d in dial]
        tokens = np.array(all_vecs)
        tokens = self.down_model.transform(tokens)
        down_vecs = []
        dial_ct = 0
        dial = []
        for toki, token in enumerate(tokens):
            if all_vec_ids[toki] == dial_ct:
                dial.append(torch.tensor(token))
            else:
                down_vecs.append(dial)
                dial = [torch.tensor(token)]
                dial_ct = all_vec_ids[toki]
        # last
        down_vecs.append(dial)
        return np.array(down_vecs)

    def simple_pca_predict(self, tensor):
        """
        tensor: B x sent
        :param tensor:
        :return:
        """
        tokens = tensor.numpy()
        tokens = self.down_model.transform(tokens)
        return tokens

    def downsample(self, vecs=None):
        """
        Downsample the data from BERT embeddings to lower dimensions
        :return:
        """
        if vecs is None:
            if self.args.mode == "train" and not self.args.eval_val:
                # train pca
                self.pca_train()
                self.save_pca_model()
            # predict tokens
            # self.dial_vecs = self.pca_predict(self.dial_vecs)
        else:
            return self.pca_predict(vecs)

    def prepare_for_finetuning(self):
        """
        Prepare data for BERT based finetuning
        https://github.com/huggingface/pytorch-transformers/tree/master/examples/lm_finetuning
        Format:
            "The scripts in this folder expect a single file as input, consisting of
            untokenized text, with one sentence per line, and one blank line between
            documents. The reason for the sentence splitting is that part of BERT's
            training involves a next sentence objective in which the model must predict
            whether two sequences of text are contiguous text from the same document
            or not, and to avoid making the task too easy, the split point between
            the sequences is always at the end of a sentence. The linebreaks
            in the file are therefore necessary to mark the points where the
            text can be split."
        :return:
        """
        self.logbook.write_message_logs("Preparing data for finetuning")
        indices = self.train_indices
        if args.mode == "test":
            indices = self.test_indices
        df = self.dialogs["true_response"]
        with open(
            "fine_tune_{}_{}.txt".format(self.args.data_name, self.args.mode), "w"
        ) as fp:
            uniq_dial_ids = list(df["dialog_id"].unique())
            for dialog_id in uniq_dial_ids:
                context = (
                    df[df.dialog_id == dialog_id]
                    .sort_values(by=["context_id"], ascending=False)["context"]
                    .values[0]
                )
                context = context.split("\n")
                response = (
                    df[df.dialog_id == dialog_id]
                    .sort_values(by=["context_id"], ascending=False)["true_response"]
                    .values[0]
                )
                dialog = context + [response]
                for utt in dialog:
                    utt = utt.replace("[CLS] ", "")
                    utt = utt.replace(" [SEP]", "")
                    sents = sent_tokenize(utt)
                    for sent in sents:
                        fp.write(sent + "\n")
                # blank line for end of doc
                fp.write("\n")
        self.logbook.write_message_logs("Done")


class DialogDataLoader(Dataset):
    """
    Dataloader for corrupt dialog and true dialogs
    """

    def __init__(
        self,
        args,
        data: Data,
        online=False,
        indices=None,
        bert_input=False,
        is_transition_fn=False,
    ):
        self.args = args
        self.data = data
        self.online = online
        self.indices = indices  # Train or Test indices
        self.bert_input = bert_input
        self.is_transition_fn = is_transition_fn

        # if args.vector_mode:
        #     if online:
        #         dialogs = [self.data.dialogs[di] for di in indices]
        #         dial_vecs = [self.data.extract_sentence_bert(dial) for dial in dialogs]
        #     else:
        #         dial_vecs = [self.data.dial_vecs[di] for di in indices]
        # else:
        # TODO: modify for RUBER later
        # self.dialogs = [self.data.dialogs[di] for di in indices]
        # self.cd = CorruptDialog(self.args, self.data, False, bert_tokenize=True)
        # self.interactions = []
        # for di, dial in enumerate(self.dialogs):
        #     dialog_id = indices[di]
        #     for i in range(1, len(dial)):
        #         self.interactions.append({
        #             'X': dial[0:i],
        #             'Y': dial[i],
        #             'dialog_id': dialog_id,
        #             'context_id': i
        #         })
        # shuffle
        self.indices = random.sample(self.indices, len(self.indices))
        # self.interactions = random.sample(self.interactions,
        #                                   len(self.interactions))

    def get_response(self, key, dialog_id, context_id, variable=True):
        if variable:
            keys = [k for k in self.data.dialogs if key in k]
            one_key = random.choice(keys)  # word_drop_0
        else:
            one_key = key
        df = self.data.dialogs[one_key]
        df = df[(df.dialog_id == dialog_id) & (df.context_id == context_id)]
        # if self.bert_input:
        #     return df["bert_" + key].values[0]
        # else:
        return df[key].values[0]  # key = word_drop

    def get_context(self, key, dialog_id, context_id, variable=True):
        if variable:
            keys = [k for k in self.data.dialogs if key in k]
            one_key = random.choice(keys)  # word_drop_0
        else:
            one_key = key
        df = self.data.dialogs[one_key]
        df = df[(df.dialog_id == dialog_id) & (df.context_id == context_id)]
        # if self.bert_input:
        #     return df["bert_" + key].values[0]
        # else:
        return df["context"].values[0]  # key = word_drop

    def get_next_response(self, key, dialog_id, context_id):
        keys = [k for k in self.data.dialogs if key in k]
        for one_key in keys:
            df = self.data.dialogs[one_key]
            df = df[(df.dialog_id == dialog_id) & (df.context_id == context_id)]
            # if self.bert_input:
            #     return df["bert_" + key].values[0]
            # else:
            yield df[key].values[0]  # key = word_drop

    def get_sents(self, item, multiple_false_responses=False, use_backtranslate=False):
        item_id = self.indices[item]
        row = self.data.dialogs["true_response"].loc[item_id]
        # if self.bert_input:
        #     context = row['bert_context']
        #     true_response = row['bert_true_response']
        # else:
        context = row["context"]
        corrupt_context = None
        true_response = row["true_response"]
        context = context.split("\n")
        dialog_id = row["dialog_id"]
        context_id = row["context_id"]
        if use_backtranslate:
            flip = random.uniform(0, 1)
            if flip > 0.5:
                true_response = self.get_response('backtranslate', dialog_id, context_id, variable=False)
        frs = []
        if self.args.corrupt_type == "only_syntax":
            for fs in only_syntax:
                frs.append(self.get_response(fs, dialog_id, context_id, variable=fs in variable_suffixes))
        elif self.args.corrupt_type == "only_semantics":
            for fs in only_semantics:
                frs.append(self.get_response(fs, dialog_id, context_id, variable=fs in variable_suffixes))
        elif self.args.corrupt_type in ["all","all_context"]:
            for fs in all_corrupt:
                frs.append(self.get_response(fs, dialog_id, context_id, variable=fs in variable_suffixes))
            if self.args.corrupt_type == "all_context":
                corrupt_context = self.get_context("corrupt_context", dialog_id, context_id, variable=True)
                frs.append(self.get_response("corrupt_context", dialog_id, context_id, variable=True))
        else:
            variable = self.args.corrupt_type in variable_suffixes
            # false_response = [self.get_response(
            #     self.args.corrupt_type, dialog_id, context_id, variable=variable
            # )]
            frs = []
            for ri, response in enumerate(self.get_next_response(self.args.corrupt_type, dialog_id, context_id)):
                if ri > self.args.num_nce - 1:
                    break
                frs.append(response)
            false_response = frs
        if multiple_false_responses:
            false_response = frs
        else:
            false_response = [random.choice(frs)]
        return context, true_response, false_response, corrupt_context

    def __getitem__(self, item):
        """
        Return single instance
        :param item:
        :return:
        """
        # inter = self.interactions[item]
        # dialog_id = inter['dialog_id']
        # context_id = inter['context_id']
        # if self.args.corrupt_type == "all":
        #     typs = ["rand_utt","drop","shuffle","model_true","model_false"]
        #     self.args.corrupt_type = random.choice(typs)
        # if self.args.corrupt_type == "rand_utt":
        #     sc = self.cd.random_clean(dialog_id=dialog_id)
        # elif self.args.corrupt_type == "drop":
        #     sc = self.cd.random_drop(self.data.dialogs[dialog_id][context_id],
        #                         drop=self.args.drop_per)
        # elif self.args.corrupt_type == "shuffle":
        #     sc = self.cd.change_word_order(self.data.dialogs[dialog_id][context_id])
        # elif self.args.corrupt_type in ["model_true", "model_false"]:
        #     sc = self.cd.get_nce_semantics(dialog_id, context_id)
        # else:
        #     raise NotImplementedError("args.corrupt_type {} not implemented".format(
        #         self.args.corrupt_type))
        X_hat = None
        multiple_false_responses = self.args.train_mode == "nce"
        context, true_response, false_responses, corrupt_context = self.get_sents(
            item, multiple_false_responses=multiple_false_responses
        )
        # tokenize X and Y
        X = [self.data.tokenizer.tokenize(sent) for sent in context]
        X = [self.data.tokenizer.convert_tokens_to_ids(sent) for sent in X]
        if corrupt_context:
            X_hat = [self.data.tokenizer.tokenize(sent) for sent in corrupt_context]
            X_hat = [self.data.tokenizer.convert_tokens_to_ids(sent) for sent in X_hat]
        Y = self.data.tokenizer.convert_tokens_to_ids(
            self.data.tokenizer.tokenize(true_response)
        )
        # if type(false_response) != str:
        #     print(context)
        #     print(true_response)
        #     print(false_response)
        assert type(false_responses) == list
        Y_hats = [
            self.data.tokenizer.convert_tokens_to_ids(self.data.tokenizer.tokenize(fr))
            for fr in false_responses
        ]
        # Y_hat = self.data.tokenizer.convert_tokens_to_ids(self.data.tokenizer.tokenize(false_response))
        if self.bert_input:
            if self.is_transition_fn:
                X = [
                    self.data.tokenizer.build_inputs_with_special_tokens(sent)
                    for sent in X
                ]
                if corrupt_context:
                    X_hat = [
                        self.data.tokenizer.build_inputs_with_special_tokens(sent)
                        for sent in X_hat
                    ]   
            else:
                # flatten
                X = [word for sent in X for word in sent]
                X = self.data.tokenizer.build_inputs_with_special_tokens(X)
                if corrupt_context:
                    X_hat = [word for sent in X_hat for word in sent]
                    X_hat = self.data.tokenizer.build_inputs_with_special_tokens(X_hat)
            Y = self.data.tokenizer.build_inputs_with_special_tokens(Y)
            Y_hats = [
                self.data.tokenizer.build_inputs_with_special_tokens(Y_hat)
                for Y_hat in Y_hats
            ]
        else:
            # flatten X
            X = [word for sent in X for word in sent]
            if corrupt_context:
                X_hat = [word for sent in X_hat for word in sent]
        # # corrupt context if needed
        # if self.args.train_mode != 'ref_score':
        #     if self.args.corrupt_context_type != 'none':
        #         inter['X_hat'] = self.cd.get_full_corrupt_context(dialog_id, len(inter['X']))
        #         X_hat = [self.data.tokenizer.tokenize(sent) for sent in inter['X_hat']]
        #         X_hat = [self.data.tokenizer.convert_tokens_to_ids(sent) for sent in X_hat]
        # assert len(X) > 0
        # assert len(Y) > 0
        # assert len(Y_hat) > 0
        # if len(Y_hats) == 1:
        #     return X, Y, Y_hats[0], None
        # else:
        return X, Y, Y_hats, X_hat

    def __len__(self):
        return len(self.indices)


class DialogDiskDataLoader(Dataset):
    """
    Serve files stored in disk
    """

    def __init__(self, args, mode, epoch=0):
        self.args = args
        self.epoch = epoch
        self.mode = mode
        # load saved
        self.dump = pkl.load(
            open(
                os.path.join(
                    self.args.exp_data_folder, "{}_epoch_{}.pkl".format(mode, epoch)
                ),
                "rb",
            )
        )

    def __len__(self):
        return len(self.dump)

    def __getitem__(self, item):
        return self.dump[item]


def vector_collate_fn(data):
    """
    Custom collate fn for vector mode
    :param data:
    :return:
    """
    X, Y, Y_hat = zip(*data)
    X, X_len = batchify(X, True)
    Y, _ = batchify(Y, True)
    Y_hat, _ = batchify(Y_hat, True)
    return X, X_len, Y, Y_hat


def id_collate_fn(data):
    """
    Custom collate fn which expects X having dialogs
    :param data:
    :return:
    """
    X, Y, Y_hat, _ = zip(*data)
    X, X_len, X_dial_len = batch_dialogs(X)
    Y, Y_len = batchify(Y, False)
    Y_hat, Y_hat_len = batchify(Y_hat, False)
    X_len = torch.from_numpy(X_len)
    Y_len = torch.from_numpy(Y_len)
    Y_hat_len = torch.from_numpy(Y_hat_len)
    return [X, X_len, X_dial_len, Y, Y_len, Y_hat, Y_hat_len]


def id_collate_nce_fn(data):
    """
    Custom collate fn which expects X having dialogs
    :param data:
    :return:
    """
    X, Y, Y_hats, _ = zip(*data)
    X, X_len, X_dial_len = batch_dialogs(X)
    Y, Y_len = batchify(Y, False)
    Y_hats, Y_hat_lens = batch_yhats(Y_hats)
    X_len = torch.from_numpy(X_len)
    Y_len = torch.from_numpy(Y_len)
    return [X, X_len, X_dial_len, Y, Y_len, Y_hats, Y_hat_lens]


def id_collate_flat_fn(data):
    """
    Custom collate fn which expects flattened X
    :param data:
    :return:
    """
    X, Y, Y_hat, _ = zip(*data)
    X, X_len = batch_words(X)
    Y, Y_len = batchify(Y, False)
    Y_hat, Y_hat_len = batchify(Y_hat, False)
    X_len = torch.from_numpy(X_len)
    Y_len = torch.from_numpy(Y_len)
    Y_hat_len = torch.from_numpy(Y_hat_len)
    return [X, X_len, Y, Y_len, Y_hat, Y_hat_len]


def id_collate_flat_nce_fn(data):
    """
    Custom collate fn which expects flattened X with NCE
    :param data:
    :return:
    """
    X, Y, Y_hats, _ = zip(*data)
    X, X_len = batch_words(X)
    Y, Y_len = batchify(Y, False)
    Y_hats, Y_hat_lens = batch_yhats(Y_hats)
    X_len = torch.from_numpy(X_len)
    Y_len = torch.from_numpy(Y_len)
    return [X, X_len, Y, Y_len, Y_hats, Y_hat_lens]

def context_collate_nce_fn(data):
    """
    Custom collate fn which expects X having dialogs
    :param data:
    :return:
    """
    X, Y, Y_hats, X_hat = zip(*data)
    X, X_len, X_dial_len = batch_dialogs(X)
    X_hat, X_hat_len, X_hat_dial_len = batch_dialogs(X_hat)
    Y, Y_len = batchify(Y, False)
    Y_hats, Y_hat_lens = batch_yhats(Y_hats)
    X_len = torch.from_numpy(X_len)
    X_hat_len = torch.from_numpy(X_hat_len)
    Y_len = torch.from_numpy(Y_len)
    return [X, X_len, X_dial_len, Y, Y_len, Y_hats, Y_hat_lens, X_hat, X_hat_len, X_hat_dial_len]

def context_collate_flat_fn(data):
    """
    Custom collate fn which expects flattened X and corrupt X_hat
    :param data:
    :return:
    """
    X, Y, Y_hat, X_hat = zip(*data)
    X, X_len = batch_words(X)
    X_hat, X_hat_len = batch_words(X_hat)
    Y, Y_len = batchify(Y, False)
    Y_hat, Y_hat_len = batchify(Y_hat, False)
    X_len = torch.from_numpy(X_len)
    Y_len = torch.from_numpy(Y_len)
    X_hat_len = torch.from_numpy(X_hat_len)
    Y_hat_len = torch.from_numpy(Y_hat_len)
    return [X, X_len, Y, Y_len, Y_hat, Y_hat_len, X_hat, X_hat_len]


def context_collate_flat_nce_fn(data):
    """
    Custom collate fn which expects flattened X with NCE and corrupt_X_hat
    :param data:
    :return:
    """
    X, Y, Y_hats, X_hat = zip(*data)
    X, X_len = batch_words(X)
    X_hat, X_hat_len = batch_words(X_hat)
    Y, Y_len = batchify(Y, False)
    Y_hats, Y_hat_lens = batch_yhats(Y_hats)
    X_len = torch.from_numpy(X_len)
    Y_len = torch.from_numpy(Y_len)
    X_hat_len = torch.from_numpy(X_hat_len)
    return [X, X_len, Y, Y_len, Y_hats, Y_hat_lens, X_hat, X_hat_len]


def id_collate_ruber_fn(data):
    """
    Custom collate fn for id mode
    :param data:
    :return:
    """
    X, Y, Y_hat, _ = zip(*data)
    # flatten X
    X = [[word for sent in dial for word in sent] for dial in X]
    X, X_len = batch_words(X)
    Y, Y_len = batchify(Y, False)
    Y_hat, Y_hat_len = batchify(Y_hat, False)
    X_len = torch.from_numpy(X_len)
    Y_len = torch.from_numpy(Y_len)
    Y_hat_len = torch.from_numpy(Y_hat_len)
    return [X, X_len, Y, Y_len, Y_hat, Y_hat_len]


def id_context_collate_fn(data):
    """
    Custom collate fn for id mode
    :param data:
    :return:
    """
    X, Y, Y_hat, X_hat = zip(*data)
    X, X_len = batch_dialogs(X)
    X_hat, X_hat_len = batch_dialogs(X_hat)
    Y, _ = batchify(Y, False)
    Y_hat, _ = batchify(Y_hat, False)
    X_len = torch.from_numpy(X_len)
    X_hat_len = torch.from_numpy(X_hat_len)
    return [X, X_len, Y, Y_hat, X_hat, X_hat_len]


class ParlAIExtractor(Data):
    """
    Extract dialog datasets from ParlAI
    """

    def __init__(self, args, logbook):
        super().__init__(args, logbook)
        self.parlai_opts = []

    def setup_args(self, parser=None):
        if parser is None:
            parser = ParlaiParser(True, True, "Display data from a task")
        parser.add_pytorch_datateacher_args()
        # Get command line arguments
        # parser.add_argument('-ne', '--num_examples', type=int, default=10)
        # parser.add_argument('-mdl', '--max_display_len', type=int, default=1000)
        # parser.add_argument('--display_ignore_fields', type=str, default='agent_reply')
        # IR Baseline arguments
        # parser.add_argument('--length_penalty', type=float, default=0.5)
        # parser.add_argument('--history_size', type=10, default=1000)
        # parser.add_argument('--use_reply', type=str, default='label')
        # parser.set_defaults(datatype='train:stream')
        return parser

    def prepare_args(self):
        """
        Prepare args
        :return:
        """
        parser = self.setup_args()
        mode = self.args.mode
        if mode == "train":
            # mode = 'train:ordered'
            mode = "train:evalmode"
        self.parlai_opts.append("-dt {}".format(mode))
        self.parlai_opts.append("-t {}".format(self.args.data_name))
        if self.args.agent == "ir":
            # IR Baseline
            self.parlai_opts.append("--length_penalty 0.5")
            self.parlai_opts.append("--history_size 10")
            self.parlai_opts.append("--use_reply label")
        elif self.args.agent == "seq2seq":
            # Seq2Seq opts
            self.parlai_opts.append(
                "-mf zoo:convai2/seq2seq/convai2_self_seq2seq_model"
            )
            self.parlai_opts.append("-m legacy:seq2seq:0")
            self.parlai_opts.append("-opt sgd")
        elif self.args.agent == "kvmemnn":
            self.parlai_opts.append("-mf zoo:convai2/kvmemnn/model")
            # self.parlai_opts.append("-nt 40")
        elif self.args.agent == "polyencoder":
            self.parlai_opts.append("-mf zoo:pretrained_transformers/model_poly/model")
            self.parlai_opts.append("--no_cuda")
        else:
            # specific model and path
            self.parlai_opts.append("-mf {}".format(self.args.mf))
        opt = parser.parse_args(" ".join(self.parlai_opts).split(" "))
        if self.args.agent == "seq2seq":
            # FIX: temporary fix for parlai issue
            opt["override"] = {
                "model": "legacy:seq2seq:0",
                "model_file": "/private/home/koustuvs/mlp/parlai_koustuvs/data/models/convai2/seq2seq/convai2_self_seq2seq_model",
            }
        if self.args.agent == "polyencoder":
            opt["override"] = {
                "fixed_candidates_path": "/private/home/koustuvs/mlp/parlai_koustuvs/data/models/pretrained_transformers/convai_trainset_cands.txt",
                "ignore_bad_candidates": True,
            }
        return opt

    def create_agent_task(self, opt):
        """
        Create ParlAI agent and task
        :param opt:
        :return:
        """
        if self.args.agent == "repeat":
            agent = RepeatLabelAgent(opt)
        elif self.args.agent == "ir":
            agent = IrBaselineAgent(opt)
        else:
            agent = create_agent(opt, requireModelExists=True)
        world = create_task(opt, agent)
        return agent, world

    def extract_interactions(self):
        """
        Extract context responses
        :return:
        """
        opt = self.prepare_args()
        agent, world = self.create_agent_task(opt)
        dialogs = []
        cur_dial = []
        true_dial = []
        contexts = []
        context_hashes = []
        model_dialogs = []
        num_dialogs = world.num_episodes()
        pb = tqdm(total=num_dialogs)
        last_true = ""
        dialog_id = 0
        context_id = 0
        data_rows = []
        while True:
            world.parley()
            for a in world.acts:
                # if personachat, remove the persona
                if self.args.data_name == "convai2" and "your persona" in a["text"]:
                    text = a["text"].split("\n")[-1]
                else:
                    text = a["text"]
                if "__SILENCE__" in text:
                    break
                if "id" in a and a["id"] == self.args.data_name:
                    if len(last_true) > 0:
                        true_dial.append(last_true)
                    true_dial.append(text)
                    if "eval_labels" in a:
                        last_true = a["eval_labels"][0]
                    else:
                        last_true = a["labels"][0]
                else:
                    cont = copy.copy(true_dial)
                    cont = " \n".join(cont)
                    agent_name = self.args.agent
                    if agent_name == "repeat":
                        agent_name = "true_response"
                    m = hashlib.md5()
                    m.update(cont.encode("utf-8"))
                    cont_hash = m.hexdigest()
                    row = {
                        "dialog_id": dialog_id,
                        "context_id": context_id,
                        "context": cont,
                        agent_name: copy.copy(text),
                        "context_hash": cont_hash,
                    }
                    data_rows.append(row)
                    # if cont_hash not in self.hash2reponses:
                    #     self.hash2reponses[cont_hash] = {}
                    # self.hash2reponses[cont_hash][self.args.agent] = copy.copy(text)
                    # if cont_hash not in self.all_hashes:
                    #     contexts.append(cont)
                    #     context_hashes.append(cont_hash)
                    #     self.all_hashes.add(cont_hash)
                    context_id += 1
            # print(world.display())
            if world.episode_done():
                if len(contexts) > 0:
                    self.interactions.append(contexts)
                    self.interaction_hashes.append(context_hashes)
                contexts = []
                context_hashes = []
                true_dial = []
                last_true = ""
                pb.update(1)
                dialog_id += 1
                context_id = 0
                # if dialog_id > 100:
                #     break
            if world.epoch_done():
                print("Epoch done")
                break
        pb.close()
        df = pd.DataFrame(data_rows)
        return df

    def extract_dialogs(self):
        # DEPRECATED
        # load data
        # parlai specific stuff
        self.args.agent = "repeat"
        opt = self.prepare_args()
        agent, world = self.create_agent_task(opt)
        dialogs = []
        dialog_hashes = []
        cur_dial = []
        num_dialogs = world.num_episodes()
        pb = tqdm(total=num_dialogs)
        while True:
            world.parley()
            for a in world.acts:
                # if personachat, remove the persona
                if (
                    self.args.data_name in ["personachat", "convai2"]
                    and "your persona" in a["text"]
                ):
                    text = a["text"].split("\n")[-1]
                else:
                    text = a["text"]
                cur_sents = sent_tokenize(text)
                cur_dial.append("[CLS] " + " [SEP] ".join(cur_sents) + " [SEP]")
            # print(world.display())
            if world.episode_done():
                dialogs.append(cur_dial)
                # calc and store hash
                m = hashlib.md5()
                for cd in cur_dial:
                    m.update(cd.encode("utf-8"))
                dialog_hashes.append(m.hexdigest())
                cur_dial = []
                pb.update(1)
            if world.epoch_done():
                print("Epoch done")
                break
        pb.close()
        self.dialogs = dialogs
        self.dialog_hashes = dialog_hashes
        self.logbook.write_message_logs("{} dialogs extracted".format(len(dialogs)))
        self.logbook.write_message_logs(
            "{} unique hashes".format(len(set(dialog_hashes)))
        )

    def extract_all_models(self):
        """
        Extract the responses from all models
        :return:
        """
        # self.load_interactions()
        models = self.args.models.split(",")
        for model in models:
            self.args.agent = model
            print(self.args.agent)
            agent_name = self.args.agent
            if agent_name == "repeat":
                agent_name = "true_response"
            df = self.extract_interactions()
            self.save_interactions(df, agent_name)

    def save_interactions(self, df, agent_name):
        """
        save
        :return:
        """
        response_file = os.path.join(
            self.args.response_file,
            "{}_{}_{}.csv".format(self.args.data_name, self.args.mode, agent_name),
        )
        df.to_csv(response_file)
        # pkl.dump({'interactions': self.interactions,
        #           'interaction_hashes': self.interaction_hashes,
        #           'hash2responses': self.hash2reponses,
        #           'all_hashes': self.all_hashes}, open(response_file, 'wb'))
        self.logbook.write_message_logs("Responses saved in {}".format(response_file))

    def load_interactions(self):
        """
        Load
        :return:
        """
        response_file = os.path.join(
            self.args.response_file,
            "{}_{}_responses.pkl".format(self.args.data_name, self.args.mode),
        )
        self.logbook.write_message_logs(
            "loading responses from {}".format(response_file)
        )
        if os.path.exists(response_file):
            dump = pkl.load(open(response_file, "rb"))
            self.interactions = dump["interactions"]
            self.hash2responses = dump["hash2responses"]
            self.all_hashes = dump["all_hashes"]
            self.interaction_hashes = dump["interaction_hashes"]
        else:
            self.logbook.write_message_logs("response file not found")
        self.model_responses = self.reformat_interactions()

    def reformat_interactions(self):
        """
        reformat the interactions to include gold standard responses
        :return:
        """
        rows = []
        self.logbook.write_message_logs(
            "num interactions : {}".format(len(self.interactions))
        )
        for di, dial in enumerate(self.interactions):
            for ci, cont in enumerate(dial):
                m = hashlib.md5()
                m.update(cont.encode("utf-8"))
                cont_hash = m.hexdigest()
                responses = self.hash2responses[cont_hash]
                responses["context"] = cont
                responses["dialog_id"] = di
                responses["context_id"] = ci
                num_cont = len(cont.split("\n"))
                responses["gold_response"] = (
                    self.dialogs[di][num_cont]
                    .replace("[CLS] ", "")
                    .replace(" [SEP]", "")
                )
                rows.append(copy.copy(responses))
        store = {}
        for row in rows:
            if row["dialog_id"] not in store:
                store[row["dialog_id"]] = {}
            rd = copy.copy(row)
            del rd["context_id"]
            del rd["dialog_id"]
            cd = {}
            for key, val in rd.items():
                if type(key) != str:
                    # necessary for alicebot responses
                    cd[key.decode("utf-8")] = val.decode("utf-8")
                else:
                    cd[key] = val
            store[row["dialog_id"]][row["context_id"]] = cd
        return store


if __name__ == "__main__":
    args = get_args()
    logbook = LogBook(vars(args))
    logbook.write_metadata_logs(vars(args))
    print("Loading {} data".format(args.data_name))
    data = ParlAIExtractor(args, logbook)
    if args.only_data:
        data.load()
        data.split_train_test()
        data.prepare_for_finetuning()
    else:
        data.extract_all_models()
        # if args.downsample:
        #     logbook.write_message_logs("Downsampling to {}".format(args.down_dim))
        #     data.downsample()
        # data.prepare_data('train')
        # data.prepare_data('test')
