"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# Compute corruptions on the given files
# Given an input file name, perform semantic or syntactic corruptions and save the raw text files
# Compute for n epochs
from args import get_args
import pandas as pd
import random
import copy
from tqdm import tqdm
from transformers import BertTokenizer

load_path = "/checkpoint/koustuvs/dialog_metric/cleaned/distilbert_lm"
import ray

ray.init()


@ray.remote
def prepare_corruptions(args, pid=0, scheme="rand_utt"):
    true_df = pd.read_csv(args.corrupt_pre + "true_response.csv")
    seq_df = pd.read_csv(args.corrupt_pre + "seq2seq.csv")
    back_trans = pd.read_csv(args.corrupt_pre + "backtranslate.csv")
    print("[{}] loaded data".format(pid))
    ## Semantic Corruption
    if scheme == "rand_utt":
        # NCE Random Utterance
        rand_utt_rows = []
        # pb_c = tqdm(total=len(true_df))
        for i, row in true_df.iterrows():
            dial_ids = list(true_df["dialog_id"].unique())
            dial_ids.remove(row["dialog_id"])
            other_dial = random.choice(dial_ids)
            sampled_resp = (
                true_df[true_df["dialog_id"] == other_dial]
                .sample(n=1)["true_response"]
                .values[0]
            )
            row = {
                "dialog_id": row["dialog_id"],
                "context_id": row["context_id"],
                "context": row["context"],
                "context_hash": row["context_hash"] if "context_hash" in row else 1,
                "rand_utt": sampled_resp,
            }
            rand_utt_rows.append(row)
            # pb_c.update(1)
        rand_utt_df = pd.DataFrame(rand_utt_rows)
        rand_utt_df.to_csv(args.corrupt_pre + "rand_utt_{}.csv".format(pid))
        # pb_c.close()
    if scheme == "model_false":
        # NCE Random Model Response
        model_false_rows = []
        # pb_c = tqdm(total=len(true_df))
        for i, row in seq_df.iterrows():
            dial_ids = list(true_df["dialog_id"].unique())
            dial_ids.remove(row["dialog_id"])
            other_dial = random.choice(dial_ids)
            sampled_resp = (
                seq_df[seq_df["dialog_id"] == other_dial]
                .sample(n=1)["seq2seq"]
                .values[0]
            )
            row = {
                "dialog_id": row["dialog_id"],
                "context_id": row["context_id"],
                "context": row["context"],
                "context_hash": row["context_hash"] if "context_hash" in row else 1,
                "model_false": sampled_resp,
            }
            model_false_rows.append(row)
            # pb_c.update(1)
        model_false_df = pd.DataFrame(model_false_rows)
        model_false_df.to_csv(args.corrupt_pre + "model_false_{}.csv".format(pid))
    if scheme == "rand_back":
        # NCE Random Backtranslation response
        rand_back_rows = []
        # pb_c = tqdm(total=len(true_df))
        for i, row in back_trans.iterrows():
            dial_ids = list(true_df["dialog_id"].unique())
            dial_ids.remove(row["dialog_id"])
            other_dial = random.choice(dial_ids)
            sampled_resp = (
                back_trans[back_trans["dialog_id"] == other_dial]
                .sample(n=1)["backtranslate"]
                .values[0]
            )
            row = {
                "dialog_id": row["dialog_id"],
                "context_id": row["context_id"],
                "context": row["context"],
                "context_hash": row["context_hash"] if "context_hash" in row else 1,
                "rand_back": sampled_resp,
            }
            rand_back_rows.append(row)
            # pb_c.update(1)
        rand_back_df = pd.DataFrame(rand_back_rows)
        rand_back_df.to_csv(args.corrupt_pre + "rand_back_{}.csv".format(pid))
    ## Syntactic Corruption
    if scheme == "word_drop":
        # NCE Random Drop
        nce_drop_rows = []
        # pb_c = tqdm(total=len(true_df))
        for i, row in true_df.iterrows():
            response = row["true_response"]
            words = response.split(" ")
            drop_word_pos = []
            for wi, word in enumerate(words):
                flip = random.uniform(0, 1)
                if flip <= args.drop_per and word not in ["[CLS]", "[SEP]"]:
                    drop_word_pos.append(wi)
            # import ipdb; ipdb.set_trace()
            response = [r for i, r in enumerate(words) if i not in drop_word_pos]
            if len(response) == 0:
                response = random.sample(words, len(words))
            if len(response) == 0:
                print("response zero")
            response = " ".join(response)
            if len(response.strip()) == 0:
                response = "word"
            row = {
                "dialog_id": row["dialog_id"],
                "context_id": row["context_id"],
                "context": row["context"],
                "context_hash": row["context_hash"] if "context_hash" in row else 1,
                "word_drop": response,
            }
            nce_drop_rows.append(row)
            # pb_c.update(1)
        nce_drop_df = pd.DataFrame(nce_drop_rows)
        nce_drop_df.to_csv(args.corrupt_pre + "word_drop_{}.csv".format(pid))
    # pb_c.close()
    if scheme == "rand_word_drop":
        # NCE Random Word Drop
        nce_rand_drop_rows = []
        # pb_c = tqdm(total=len(true_df))
        for i, row in true_df.iterrows():
            dial_ids = list(true_df["dialog_id"].unique())
            dial_ids.remove(row["dialog_id"])
            other_dial = random.choice(dial_ids)
            response = (
                true_df[true_df["dialog_id"] == other_dial]
                .sample(n=1)["true_response"]
                .values[0]
            )
            words = response.split(" ")
            drop_word_pos = []
            for wi, word in enumerate(words):
                flip = random.uniform(0, 1)
                if flip <= args.drop_per and word not in ["[CLS]", "[SEP]"]:
                    drop_word_pos.append(wi)
            # import ipdb; ipdb.set_trace()
            response = [r for i, r in enumerate(words) if i not in drop_word_pos]
            if len(response) == 0:
                response = random.sample(words, len(words))
            if len(response) == 0:
                print("response zero")
            response = " ".join(response)
            if len(response.strip()) == 0:
                response = "word"
            row = {
                "dialog_id": row["dialog_id"],
                "context_id": row["context_id"],
                "context": row["context"],
                "context_hash": row["context_hash"] if "context_hash" in row else 1,
                "rand_word_drop": response,
            }
            nce_rand_drop_rows.append(row)
            # pb_c.update(1)
        nce_rand_drop_df = pd.DataFrame(nce_rand_drop_rows)
        nce_rand_drop_df.to_csv(args.corrupt_pre + "rand_word_drop_{}.csv".format(pid))
    if scheme == "word_order":
        # NCE Change word order
        nce_order_rows = []
        # pb_c = tqdm(total=len(true_df))
        for i, row in true_df.iterrows():
            response = row["true_response"]
            words = response.split(" ")
            response = random.sample(words, len(words))
            response = " ".join(response)
            row = {
                "dialog_id": row["dialog_id"],
                "context_id": row["context_id"],
                "context": row["context"],
                "context_hash": row["context_hash"] if "context_hash" in row else 1,
                "word_order": response,
            }
            nce_order_rows.append(row)
            # pb_c.update(1)
        nce_order_df = pd.DataFrame(nce_order_rows)
        nce_order_df.to_csv(args.corrupt_pre + "word_order_{}.csv".format(pid))
    # pb_c.close()
    if scheme == "word_repeat":
        # choose a random word in the sentence and start repeating from that word
        # in order to mimic "i have have have ..."  common seq2seq behaviour
        # NCE Change word order
        nce_repeat_rows = []
        # pb_c = tqdm(total=len(true_df))
        for i, row in true_df.iterrows():
            response = row["true_response"]
            words = response.split(" ")
            repeat_word = random.choice(words[:2])
            repeat_word_indx = words.index(repeat_word)
            response = words[:repeat_word_indx]
            response = response + [repeat_word] * (len(words) - len(response))
            response = " ".join(response)
            row = {
                "dialog_id": row["dialog_id"],
                "context_id": row["context_id"],
                "context": row["context"],
                "context_hash": row["context_hash"] if "context_hash" in row else 1,
                "word_repeat": response,
            }
            nce_repeat_rows.append(row)
            # pb_c.update(1)
        nce_repeat_df = pd.DataFrame(nce_repeat_rows)
        nce_repeat_df.to_csv(args.corrupt_pre + "word_repeat_{}.csv".format(pid))
    ## Corrupting the context
    if scheme == "corrupt_context":
        # corrupt the context entirely
        # NCE Change word order
        tokenizer = BertTokenizer.from_pretrained(load_path)

        def get_sent_len(context):
            X = [tokenizer.tokenize(sent) for sent in context]
            X = [tokenizer.convert_tokens_to_ids(sent) for sent in X]
            X = [word for sent in X for word in sent]
            X = tokenizer.build_inputs_with_special_tokens(X)
            return len(X)

        nce_cont_cor_rows = []
        context_sents = [row["context"].split("\n") for i, row in true_df.iterrows()]
        context_sents = [y for x in context_sents for y in x]
        # pb_c = tqdm(total=len(true_df))
        for i, row in true_df.iterrows():
            context = row["context"]
            context_len = len(row["context"].split("\n"))
            while True:
                random_context = random.sample(context_sents, context_len)
                if get_sent_len(random_context) < 512:
                    break
            response = row["true_response"]
            row = {
                "dialog_id": row["dialog_id"],
                "context_id": row["context_id"],
                "context": "\n".join(random_context),
                "context_hash": row["context_hash"] if "context_hash" in row else 1,
                "corrupt_context": response,
            }
            nce_cont_cor_rows.append(row)
            # pb_c.update(1)
        nce_cont_cor_df = pd.DataFrame(nce_cont_cor_rows)
        nce_cont_cor_df.to_csv(args.corrupt_pre + "corrupt_context_{}.csv".format(pid))
    # pb_c.close()
    return "[{}] {} done".format(pid, scheme)


if __name__ == "__main__":
    args = get_args()
    schemes = [
        # "rand_utt",
        # "model_false",
        # "rand_back",
        # "word_order",
        # "word_drop",
        # "word_repeat",
        "corrupt_context",
    ]
    # schemes = [
    #     "word_repeat",
    # ]
    futures = [
        prepare_corruptions.remote(args, i, scheme)
        for i in range(args.corrupt_ne)
        for scheme in schemes
    ]
    print(ray.get(futures))
