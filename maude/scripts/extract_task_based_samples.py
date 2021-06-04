"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import os
import numpy as np
import math
import argparse
import csv
import os
from copy import deepcopy
from nltk.tokenize import sent_tokenize

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=str)
parser.add_argument("--increment", type=int, default=4)
parser.add_argument("--destination", type=str, default="./")
parser.add_argument("--type", type=str, default="train")

args = parser.parse_args()

if "s2s" in args.sample:
    model = "seq2seq"
else:
    model = "hred"

if "mwoz" in args.sample:
    dataset = "mwoz"
else:
    dataset = "frames"

csvwriter_model = csv.writer(
    open(
        os.path.join(
            args.destination, dataset + "_" + args.type + "_" + model + ".csv"
        ),
        "w",
    )
)
csvwriter_true = csv.writer(
    open(
        os.path.join(
            args.destination, dataset + "_" + args.type + "_true_response.csv"
        ),
        "w",
    )
)

csvwriter_model.writerow(["index", "dialog_id", "context_id", "context", model])

csvwriter_true.writerow(
    ["index", "dialog_id", "context_id", "context", "true_response"]
)


def generateSample(file_loc):
    fp = open(file_loc)
    D = fp.readlines()
    i = 0
    dialog_id = 0
    context_id = 0
    index = 0
    prev_con = ""
    while i < len(D):
        con = D[i].split()
        tar = D[i + 2].split()
        mod = D[i + 1].split()
        if "<eos>" in mod:
            mod = " ".join(mod[1 : mod.index("<eos>")])
        else:
            mod = " ".join(mod[1:])
        if "<eos>" in tar:
            tar = " ".join(tar[1 : tar.index("<eos>")])
        else:
            tar = " ".join(tar[1:])
        con = " ".join(con[1:])
        for tok in ["<go>", "<pad>", "<eos>"]:
            con = con.replace(tok, "")
            tar = tar.replace(tok, "")
            mod = mod.replace(tok, "")
        if prev_con != "":
            if len(prev_con) > len(con) + 2:
                dialog_id += 1
                context_id = 0
        prev_con = deepcopy(con)

        con = "\n".join(sent_tokenize(con))
        csvwriter_true.writerow([str(index), str(dialog_id), str(context_id), con, tar])
        csvwriter_model.writerow(
            [str(index), str(dialog_id), str(context_id), con, mod]
        )
        context_id += 1
        index += 1
        i += args.increment


if __name__ == "__main__":
    generateSample(args.sample)
    print("done")
