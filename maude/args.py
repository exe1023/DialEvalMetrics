"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
from test_tube import HyperOptArgumentParser
import argparse


def get_args(command=None):
    # parser = argparse.ArgumentParser()
    parser = HyperOptArgumentParser(strategy="random_search", add_help=False)
    # Analysis args
    # parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help="bert model name")
    parser.add_argument(
        "--bert_model",
        default="distilbert-base-uncased",
        type=str,
        help="bert model name",
    )
    parser.add_argument(
        "--emb_file",
        default="~/checkpoint/bert_vectors/",
        type=str,
        help="location of embedding file",
    )
    parser.add_argument(
        "--data_loc",
        default="~/checkpoint/dialog_metric/convai2_data/",
        type=str,
        help="location of data dump",
    )
    parser.add_argument(
        "--data_name", default="convai2", type=str, help="convai2/cornell_movie"
    )
    parser.add_argument(
        "--tok_file", default="na", type=str, help="tokens and word dict file"
    )
    parser.add_argument(
        "--pca_file", default="na", type=str, help="pca saved weights file"
    )
    parser.opt_list(
        "--learn_down",
        default=False,
        action="store_true",
        options=[True, False],
        tunable=False,
    )
    parser.opt_list(
        "--fix_down",
        default=False,
        action="store_true",
        options=[True, False],
        tunable=False,
    )
    parser.add_argument(
        "--trained_bert_suffix",
        default="ep_10_lm",
        type=str,
        help="folder to look for trained bert",
    )
    parser.add_argument("--tc", default=False, action="store_true")
    parser.opt_list(
        "--downsample",
        default=True,
        action="store_true",
        options=[True, False],
        tunable=False,
    )
    parser.opt_list(
        "--down_dim", type=int, default=300, options=[100, 300, 400], tunable=False
    )
    parser.add_argument("--load_fine_tuned", default=True, action="store_true")
    # parser.add_argument("--fine_tune_model", default="~/checkpoint/dialog_metric/cleaned/bert_lm",type=str)
    parser.add_argument(
        "--fine_tune_model",
        default="~/checkpoint/dialog_metric/cleaned/distilbert_lm",
        type=str,
    )
    # Experiment ID
    parser.add_argument("--id", default="ruber_bs", type=str)
    # Model training args
    parser.add_argument("--device", default="cuda", type=str, help="cuda/cpu")
    parser.add_argument(
        "--model",
        default="models.TransitionPredictorMaxPoolLearnedDownsample",
        type=str,
        help="full model name path",
    )
    parser.opt_list(
        "--optim",
        default="adam,lr=0.0001",
        type=str,
        help="optimizer",
        options=["adam,lr=0.001", "adam,lr=0.01", "adam,lr=0.0001"],
        tunable=False,
    )
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("--margin", default=0.5, type=float, help="margin")
    parser.add_argument(
        "--train_mode",
        default="ref_score",
        type=str,
        help="ref_score/cont_score/all/nce",
    )
    parser.add_argument(
        "--num_nce", type=int, default=5, help="number of nce samples per scheme"
    )
    parser.add_argument(
        "--model_save_dir",
        default="~/checkpoint/dialog_metric/",
        type=str,
        help="model save dir",
    )
    parser.add_argument(
        "--model_load_path",
        default="~/checkpoint/dialog_metric/",
        type=str,
        help="if there is a need of different load path",
    )
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument(
        "--load_model",
        default=False,
        action="store_true",
        help="load model from previous checkpoint",
    )
    parser.add_argument(
        "--logger_dir",
        default="./",
        type=str,
        help="log directory (must be created)",
    )
    parser.add_argument("--log_interval", default=100, type=int, help="log interval")
    parser.add_argument(
        "--watch_model", default=False, action="store_true", help="wandb watch model"
    )
    parser.add_argument(
        "--vector_mode",
        default=True,
        action="store_true",
        help="if false, train with word representations",
    )
    parser.add_argument(
        "--remote_logging",
        default=False,
        action="store_true",
        help="wandb remote loggin on or off",
    )
    parser.add_argument("--wandb_project", default="dialog-metric", type=str)
    parser.add_argument("--bidirectional", default=False, action="store_true")
    parser.add_argument("--dataloader_threads", default=8, type=int)
    parser.add_argument("--exp_data_folder", default="na", help="exp data folder")
    parser.add_argument(
        "--num_workers", default=4, type=int, help="dataloader num workers"
    )
    parser.opt_list(
        "--clip",
        default=0.5,
        type=float,
        help="gradient clipping",
        options=[0.0, 0.5, 1.0],
        tunable=False,
    )
    parser.opt_list(
        "--dropout",
        default=0.2,
        type=float,
        help="gradient clipping",
        options=[0.0, 0.2],
        tunable=False,
    )
    parser.opt_list(
        "--decoder_hidden",
        default=200,
        type=int,
        help="decoder hidden values",
        options=[100, 200, 500, 700],
        tunable=False,
    )
    parser.add_argument(
        "--gpus", type=str, default="-1", help="how many gpus to use in the node"
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="if true, set debug modes"
    )
    ## Evaluation args
    parser.add_argument(
        "--corrupt_type",
        default="rand_utt",
        type=str,
        help="all/word_drop/word_order/word_repeat/rand_utt/model_false/rand_back/only_semantics/only_syntax/context_corrupt",
    )
    parser.add_argument(
        "--corrupt_context_type",
        default="rand",
        type=str,
        help="rand/drop/shuffle/model_true/model_false/progress/none",
    )
    parser.add_argument("--drop_per", default=0.50, type=float, help="drop percentage")
    parser.add_argument(
        "--eval_val", default=False, action="store_true", help="only eval val set"
    )
    parser.add_argument(
        "--model_response_pre",
        default="na",
        type=str,
        help="model response file prefix",
    )
    parser.add_argument(
        "--load_model_responses",
        default=True,
        action="store_true",
        help="load model responses",
    )
    parser.add_argument(
        "--corrupt_model_names",
        default="seq2seq",
        type=str,
        help="comma separated models",
    )
    parser.add_argument(
        "--restore_version",
        default=-1,
        type=int,
        help="if > -1, restore training from the given version",
    )

    # Baseline args
    parser.add_argument("--train_baseline", default="na", help="ruber/bilstm", type=str)
    ## RUBER
    parser.add_argument(
        "--word2vec_context_size",
        default=3,
        type=int,
        help="context size for word2vec training",
    )
    parser.add_argument(
        "--word2vec_embedding_dim", default=300, type=int, help="embedding dim"
    )
    parser.add_argument(
        "--word2vec_epochs", default=100, type=int, help="word2vec training epochs"
    )
    parser.add_argument(
        "--word2vec_out",
        default="~/checkpoint/dialog_metric/ruber/w2v.pt",
        type=str,
        help="word2vec output location",
    )
    parser.add_argument("--word2vec_lr", default=0.001, type=float, help="word2vec lr")
    parser.add_argument("--word2vec_batchsize", default=512, type=int)
    parser.add_argument(
        "--ruber_ref_pooling_type", default="max_min", type=str, help="max_min/avg"
    )
    parser.add_argument(
        "--ruber_unref_pooling_type", default="max", type=str, help="max/mean"
    )
    parser.add_argument(
        "--ruber_load_emb", action="store_true", help="load trained word2vec"
    )
    parser.add_argument(
        "--ruber_lstm_dim", default=300, type=int, help="dimensions of ruber encoder"
    )
    parser.add_argument(
        "--ruber_mlp_dim", default=200, type=int, help="dimensions of ruber encoder"
    )
    parser.add_argument(
        "--ruber_dropout", default=0.1, type=float, help="ruber dropout"
    )
    parser.add_argument("--num_words", default=-1, type=int)

    ## Data collection args
    parser.add_argument(
        "--agent", type=str, default="kvmemnn", help="repeat/ir/seq2seq"
    )
    parser.add_argument("--mode", type=str, default="train", help="train/test/valid")
    parser.add_argument(
        "--models",
        type=str,
        default="seq2seq,repeat",
        help="comma separated model values",
    )
    parser.add_argument(
        "--response_file", type=str, default="~/checkpoint/dialog_metric/"
    )
    parser.add_argument(
        "--mf",
        type=str,
        default="/checkpoint/parlai/zoo/convai2/seq2seq_naacl2019_abibaseline/model",
        help="only for special cases",
    )
    parser.add_argument(
        "--only_data",
        action="store_true",
        default=False,
        help="only extract and store dialog data",
    )

    ## SLURM args
    parser.add_argument(
        "--slurm_log_path",
        type=str,
        default="~/checkpoint/dialog_metrics/ckpt/",
        help="slurm log path",
    )
    parser.add_argument(
        "--per_experiment_nb_gpus", type=int, default=1, help="number of gpus"
    )
    parser.add_argument(
        "--per_experiment_nb_cpus", type=int, default=16, help="number of cpus"
    )
    parser.add_argument(
        "--nb_gpu_nodes", type=int, default=1, help="number of gpu nodes"
    )
    parser.add_argument("--job_time", type=str, default="23:59:00", help="time")
    parser.add_argument("--gpu_type", type=str, default="volta", help="gpu type")
    parser.add_argument(
        "--gpu_partition", type=str, default="learnfair", help="gpu type"
    )
    parser.add_argument(
        "--nb_hopt_trials",
        type=int,
        default=1,
        help="how many grid search trials to run",
    )
    parser.add_argument("--train_per_check", type=float, default=1.0)
    parser.add_argument("--val_per_check", type=float, default=1.0)
    parser.add_argument("--test_per_check", type=float, default=1.0)
    parser.add_argument(
        "--use_cluster",
        action="store_true",
        default=False,
        help="activate cluster mode",
    )
    ## Inference args
    parser.add_argument("--model_name", type=str, default="na", help="model name")
    parser.add_argument(
        "--model_version", type=str, default="version_0", help="model version"
    )
    parser.add_argument("--use_ddp", action="store_true", default=False)
    parser.add_argument("--human_eval", action="store_true", default=False)
    parser.add_argument(
        "--human_eval_file",
        type=str,
        default="~/checkpoint/dialog_metric/controllable_dialogs.csv",
    )
    parser.add_argument("--results_file", type=str, default="test_results.jsonl")
    ## Corruption args
    parser.add_argument(
        "--corrupt_pre",
        type=str,
        default="~/checkpoint/dialog_metric/convai2_data/convai2_test_",
    )
    parser.add_argument("--corrupt_ne", type=int, default=1)
    parser.add_argument("--test_suffix", type=str, default="true_response")
    parser.add_argument("--test_column", type=str, default="true_response")

    if command:
        return parser.parse_args(command.split(" "))
    else:
        return parser.parse_args()
