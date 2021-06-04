"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# File to run various inferences
import sys
import os

sys.path.append(os.getcwd())
import torch
from args import get_args
from logbook.logbook import LogBook
from codes.models import TransitionPredictorMaxPool
from backup.corrupt import CorruptDialog
from data import ParlAIExtractor, id_context_collate_fn
from tqdm import tqdm
import pickle as pkl
from pytorch_lightning import Trainer
from codes.models import TransitionPredictorMaxPool
from codes.baseline_models import RuberUnreferenced, InferSent, BERTNLI
import pandas as pd
import numpy as np
import os.path
import glob
import json
from utils import batchify, batch_dialogs, batch_words


def get_dial_tokens(test_data, X):
    X = [test_data.tokenizer.tokenize(sent) for sent in X]
    X = [test_data.tokenizer.convert_tokens_to_ids(sent) for sent in X]
    return X


def test_trainer(args, model):
    trainer = Trainer(
        gpus=-1,
        show_progress_bar=args.use_cluster == False,
        row_log_interval=10,
        log_save_interval=1,
        distributed_backend="dp",  # if args.debug else 'ddp',
        max_nb_epochs=1,
        # amp_level='O2', use_amp=True,
    )
    trainer.test(model)


def test_single(
    args, model, df_name, response_row="response", save_eval=False,
):
    """
    Evaluate the model on a given dataset
    This would be a single batched inference
    """
    df = pd.read_csv(df_name)
    if args.debug:
        # Choose a small sample
        print("Debug mode, choosing only 10 examples...")
        df = df.sample(10)
    pb = tqdm(total=len(df))
    test_dl = model.get_dataloader(mode="test", datamode="test")
    all_scores = []
    for i, row in df.iterrows():
        context = row["context"]
        response = row[response_row]
        context = context.split("\n")
        X = [model.test_data.tokenizer.tokenize(sent) for sent in context]
        X = [model.test_data.tokenizer.convert_tokens_to_ids(sent) for sent in X]
        Y = model.test_data.tokenizer.convert_tokens_to_ids(
            model.test_data.tokenizer.tokenize(response)
        )
        X_dial_len = None
        if model.bert_input:
            if model.is_transition_fn:
                X = [
                    model.test_data.tokenizer.build_inputs_with_special_tokens(sent)
                    for sent in X
                ]
                X, X_len, X_dial_len = batch_dialogs([X])
            else:
                # flatten
                X = [word for sent in X for word in sent]
                X = model.test_data.tokenizer.build_inputs_with_special_tokens(X)
                X, X_len = batch_words([X])
            Y = model.test_data.tokenizer.build_inputs_with_special_tokens(Y)
        else:
            X = [word for sent in X for word in sent]
            X, X_len = batch_words([X])
        Y, Y_len = batchify([Y], False)
        X_len = torch.from_numpy(X_len)
        Y_len = torch.from_numpy(Y_len)
        if model.is_transition_fn:
            score = model(X, X_len, X_dial_len, Y, Y_len)
        else:
            score = model(X, X_len, Y, Y_len)
        df.at[i, "{}_score_{}".format(args.id, args.model_version)] = score.item()
        all_scores.append(score)
        pb.update(1)
    pb.close()
    mean = np.mean(all_scores)
    std = np.std(all_scores)
    result = "> {} : mean score : {}, std: {}".format(response_row, mean, std)
    cell_format = "{}+/-{}".format(mean, std)
    with open(args.results_file, "a",) as fp:
        res_d = {
            "id": args.id,
            "response_row": response_row,
            "model_version": args.model_version,
            "test_file": df_name,
            "score_mean": str(mean),
            "score_std": str(std),
            "formatted": "{:0.2f}+/-{:0.2f}".format(mean, std),
        }
        fp.write(json.dumps(res_d) + "\n")
    print(result)
    if save_eval:
        print("saving ...")
        df.to_csv(df_name)


def test_context_corruption(args, test_data, model, only_context=False):
    # single non-batched evaluation, may take time
    mode = "ref_score"
    if only_context:
        mode = "cont_score"
    cd = CorruptDialog(args, test_data, False, bert_tokenize=True)
    progress_scores = []
    pb = tqdm(total=len(test_data.dialogs))
    for dial_id, dial in enumerate(test_data.dialogs):
        # only get the last one
        dial_scores = []
        X = dial[0:-1]
        Y = dial[-1]
        X = get_dial_tokens(test_data, X)
        Y = test_data.tokenizer.convert_tokens_to_ids(test_data.tokenizer.tokenize(Y))
        for X_hat in cd.next_corrupt_context_model(dial_id, len(X)):
            X_h = get_dial_tokens(test_data, X_hat)
            batch = id_context_collate_fn([(X, Y, Y, X_h)])
            bX, bX_len, bY, bY_hat, bX_h, bX_h_len = batch
            pred_true = model.forward(bX, bX_len, bY, mode=mode)
            pred_false = model.forward(bX_h, bX_h_len, bY, mode=mode)
            dial_scores.append((pred_true.item(), pred_false.item()))
        progress_scores.append(dial_scores)
        pb.update(1)
    pb.close()
    pkl.dump(progress_scores, open(experiment_path + "/dial_scores_context.pkl", "wb"))


if __name__ == "__main__":
    args = get_args()
    logbook = LogBook(vars(args))
    logbook.write_metadata_logs(vars(args))
    ## load data
    logbook.write_message_logs("init loading data for testing")
    # args.mode = 'test'
    # test_data = ParlAIExtractor(args, logbook)
    # test_data.load()
    # to get the correct paths
    # args = test_data.args
    experiment_path = "{}/{}/lightning_logs/version_{}".format(
        args.model_save_dir, args.id, args.model_version
    )
    model_save_path = "{}/checkpoints/*.ckpt".format(experiment_path)
    all_saved_models = glob.glob(model_save_path)
    model_save_path = all_saved_models[0]
    tag_path = "{}/meta_tags.csv".format(experiment_path)
    # load model
    model_module = None
    if len(args.train_baseline) > 0 and args.train_baseline != "na":
        if args.train_baseline == "ruber":
            model_module = RuberUnreferenced
        elif args.train_baseline == "infersent":
            model_module = InferSent
        elif args.train_baseline == "bertnli":
            model_module = BERTNLI
    else:
        model_module = TransitionPredictorMaxPool

    # Clean paths which have been left over from training
    meta_tags = pd.read_csv(tag_path)
    meta_tags.at[(meta_tags.key == "logger_dir"), "value"] = "logs/"
    for key in [
        "fine_tune_model",
        "data_loc",
        "model_save_dir",
        "model_load_path",
        "response_file",
    ]:
        orig_val = meta_tags[meta_tags.key == key].iloc[0]["value"]
        if len(str(orig_val)) > 0:
            val = str(orig_val).split("dialog_metric/")
            if len(val) > 0:
                val = val[-1]
            else:
                val = ""
            meta_tags.at[(meta_tags.key == key), "value"] = val
    meta_tags.to_csv(tag_path, index=False)

    # load metrics
    model = model_module.load_from_metrics(
        weights_path=model_save_path, tags_csv=tag_path
    )
    # Reset some parameters for easy inference
    model.set_hparam("corrupt_type", args.corrupt_type)
    # model.hparams.corrupt_type = args.corrupt_type
    # set model on evaluation mode
    model.preflight_steps()
    model.eval()
    model.freeze()

    # test_context_corruption(args, test_data, model)
    # test_trainer(args, model)
    if args.human_eval:
        test_file = args.human_eval_file
        test_single(args, model, test_file, save_eval=True)
    else:
        #test_file = args.corrupt_pre + args.test_suffix + ".csv"
        test_file = args.corrupt_pre + ".csv"
        if os.path.exists(test_file) and os.path.isfile(test_file):
            print("Testing file : {}".format(test_file))
            test_single(args, model, test_file, response_row=args.test_column, save_eval=True)
        else:
            raise FileNotFoundError("File {} not found".format(test_file))
