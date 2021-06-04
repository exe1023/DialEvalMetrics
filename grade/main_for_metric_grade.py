import os
import sys
sys.path.append('./config')
import argparse
import functools
import importlib
import logging
from typing import Any

import random
import numpy as np
np.set_printoptions(threshold = np.inf)
from time import time
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import texar.torch as tx
from tqdm import tqdm

from utils.main_utils import *



parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-model", default="config_model_grade",
    help="Configuration of the model.")
parser.add_argument(
    "--config-data", default="config_data_for_metric", 
    help="Configuration of the dataset.")
parser.add_argument(
    "--model-file", default='',
    help="Configuration of the network")
parser.add_argument(
    '--gpu', type=str, default='4',
    help="default gpu to load model and data.")
parser.add_argument(
    '--devices_id', type=str, default='4',
    help="gpu to load model and data.")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to a model checkpoint (including bert modules) to restore from.")
parser.add_argument(
    "--do-metrics", action="store_true",
    help="Whether to get metric scores on the test set.")
parser.add_argument(
    "--non_reduced_results_path",
    type=str,
    required=True,
    default=None,
    help="save non_reduced metrics scores")
parser.add_argument(
    "--reduced_results_path",
    type=str,
    required=True,
    default=None,
    help="save reduced metrics scores")
parser.add_argument(
    "--eval-metric-name",
    type=str,
    required=True,
    default=None,
    help="metric name")
parser.add_argument(
    "--eval-dataset-name",
    type=str,
    required=True,
    default=None,
    help="dataset name")
parser.add_argument(
    "--hyp-format",
    type=str,
    required=True,
    default=None,
    help="hyp file prefix")
parser.add_argument(
    "--ctx-format",
    type=str,
    default=None,
    help="ctx file prefix")
parser.add_argument(
    "--dialog-model-name", type=str, default='percvae',
    help="dialog model name")
parser.add_argument(
    "--dataset_dir",
    help="dataset dir for loading graph massage",
    default='./data/DailyDialog',
    type=str)
parser.add_argument(
    "--unlimit_hop",
    type=int,
    required=True,
    default='20',
    help="hop for unreachable")
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    default='GRADE_K2_N10_N10',
    help="Grade Version")
args = parser.parse_args()


config_data: Any = importlib.import_module(args.config_data)
config_model: Any = importlib.import_module(args.config_model)
model_file: Any = importlib.import_module(args.model_file)
net=model_file.GRADE


def main():
    output_info = 'Start to compute metric score [metric_name: {}, dialog_model: {}, dataset: {}, hyp_format: {}, ctx_format: {}]'.format(
    args.eval_metric_name, args.dialog_model_name, args.eval_dataset_name, args.hyp_format, args.ctx_format)
    print('-' * len(output_info))
    print(output_info)
    print('-' * len(output_info))

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # Loads data
    print("LOADING DATA....")
    test_dataset = tx.data.MultiAlignedData(hparams=config_data.test_hparam,
                                      device=device)
    embedding_init_value = test_dataset.embedding_init_value

    vocab2id, id2vocab = build_vocab_id(os.path.join(args.dataset_dir, "keyword.vocab"))
    pair_hops = load_tuples_hops(os.path.join(args.dataset_dir, "dialog_keyword_tuples_multiGraph.hop")) 
    if args.model_name == 'GRADE_K2_N10_N10':
        oneHop_mean_embedding_dict, twoHop_mean_embedding_dict = load_hop_mean_embedding(os.path.join(args.dataset_dir, '1st_hop_nr10.embedding'), \
            os.path.join(args.dataset_dir, '2nd_hop_nr10.embedding'))
    elif args.model_name == 'GRADE_K2_N20_N20':
        oneHop_mean_embedding_dict, twoHop_mean_embedding_dict = load_hop_mean_embedding(os.path.join(args.dataset_dir, '1st_hop_nr20.embedding'), \
            os.path.join(args.dataset_dir, '2nd_hop_nr20.embedding'))

    print("Finish loading DATA....")
    iterator = tx.data.DataIterator(
        {"test": test_dataset}
    )
    num_test_data = config_data.num_test_data

    # Builds net
    model = net(args, config_model, config_data, embedding_init_value, device)
    if device != 'cpu':
        devices_ids= [int(i) for i in args.devices_id.split(',')] 
        model = nn.DataParallel(model, device_ids=devices_ids, output_device=devices_ids[0])
    cudnn.benchmark = True
    model = model.to(device)

    @torch.no_grad()
    def _do_metrics():
        
        iterator.switch_to_dataset("test")
        model.eval()

        auto_scores = []
        reduced_metrics = {}
        non_reduced_metrics = {}

        # save scores results to files
        for batch_id, batch in tqdm(enumerate(iterator)):
            pair_1_input_ids_raw_text = batch["pair_1_input_ids_raw_text"]
            pair_1_segment_ids_raw_text = batch["pair_1_segment_ids_raw_text"]
            pair_1_input_length_raw_text = (1 - (pair_1_input_ids_raw_text == 0).int()).sum(dim=1)
            pair_1_input_mask_raw_text = batch['pair_1_input_mask_raw_text']
            pair_1_input_ids_Keywords = batch["keyword_pair_1_text_ids"]
            pair_1_input_length_Keywords = batch["keyword_pair_1_length"]
            pair_1_input_ids_ctxKeywords = batch["ctx_keyword_pair_1_text_ids"]
            pair_1_input_ids_repKeywords = batch["rep_keyword_pair_1_text_ids"]
            if args.model_name == 'GRADE_K2_N10_N10' or args.model_name =='GRADE_K2_N20_N20':
                pair_1_batched_adjs, pair_1_batch_onehop_embedding_matrix, pair_1_batch_twohop_embedding_matrix = \
                    get_adjs1(oneHop_mean_embedding_dict, twoHop_mean_embedding_dict, pair_1_input_ids_Keywords, pair_1_input_ids_ctxKeywords, \
                    pair_1_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, args.unlimit_hop)
            else:
                pair_1_batched_adjs = get_adjs2(pair_1_input_ids_Keywords, pair_1_input_ids_ctxKeywords, \
                    pair_1_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, args.unlimit_hop)
                pair_1_batch_onehop_embedding_matrix, pair_1_batch_twohop_embedding_matrix=None, None

            scores = model('metric', 
                pair_1_input_ids_raw_text=pair_1_input_ids_raw_text,
                pair_1_input_length_raw_text=pair_1_input_length_raw_text,
                pair_1_segment_ids_raw_text=pair_1_segment_ids_raw_text,
                pair_1_input_mask_raw_text=pair_1_input_mask_raw_text,
                pair_1_batched_adjs=pair_1_batched_adjs,
                pair_1_input_ids_Keywords=pair_1_input_ids_Keywords,
                pair_1_input_length_Keywords=pair_1_input_length_Keywords,
                pair_1_batch_onehop_embedding_matrix=pair_1_batch_onehop_embedding_matrix,
                pair_1_batch_twohop_embedding_matrix=pair_1_batch_twohop_embedding_matrix,
                SCORES=True)
            auto_scores.extend(scores.data.cpu().numpy().tolist())
        
        score = np.round(np.mean(auto_scores),4)
        reduced_metrics[args.eval_metric_name] = score
        auto_scores = np.squeeze(auto_scores, 1).tolist()
        non_reduced_metrics[args.eval_metric_name] = auto_scores

        return reduced_metrics, non_reduced_metrics


    if args.checkpoint:
        #ckpt = torch.load(args.checkpoint, map_location='cuda:0')
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])

    if args.do_metrics:
        print("=============Start to get metric scores=============")
        maybe_create_file('/'.join(args.non_reduced_results_path.split('/')[:-1]))

        reduced_metrics, non_reduced_metrics = _do_metrics()
        print_evaluation_results(reduced_metrics)
        save_evaluation_results(args.non_reduced_results_path, args.reduced_results_path, non_reduced_metrics, reduced_metrics)
        print('Done.\n')


if __name__ == "__main__":
    main()
