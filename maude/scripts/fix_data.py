"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
## fix data for length

import glob
import pandas as pd
from transformers import BertTokenizer

load_path = "/checkpoint/koustuvs/dialog_metric/cleaned/distilbert_lm"
import ray

ray.init()

data_loc = "/checkpoint/koustuvs/dialog_metric/frames_data/"


def get_long_sents(df):
    long_sents = []
    tokenizer = BertTokenizer.from_pretrained(load_path)
    for i, row in df.iterrows():
        context = row["context"]
        context = context.split("\n")
        X = [tokenizer.tokenize(sent) for sent in context]
        X = [tokenizer.convert_tokens_to_ids(sent) for sent in X]
        X = [word for sent in X for word in sent]
        X = tokenizer.build_inputs_with_special_tokens(X)
        if len(X) >= 512:
            long_sents.append(i)
    return long_sents


@ray.remote
def process_data(key, df, save_loc, mode="train"):
    if mode in key:
        print("getting long sents for {}".format(key))
        long_sents = get_long_sents(df)
        print("got {} long sents".format(len(long_sents)))
        if len(long_sents) > 0:
            print("removing ...")
            df.drop(long_sents, inplace=True)
            df.to_csv(save_loc)
            print("saved in {}".format(save_loc))
    else:
        print("skipping {}".format(key))
    return "done {}".format(key)


if __name__ == "__main__":
    alf = glob.glob(data_loc + "*.csv")
    data_d = {}
    mode = "test"
    for fl in alf:
        if mode in fl:
            last_name = fl.split("/")[-1].split(".csv")[0]
            data_d[last_name] = {"df": pd.read_csv(fl), "key": "", "loc": fl}
    print("loaded data, {} dataframes".format(len(data_d)))
    args = [(key, data["df"], data["loc"]) for key, data in data_d.items()]
    futures = [process_data.remote(item[0], item[1], item[2], mode) for item in args]
    print(ray.get(futures))
