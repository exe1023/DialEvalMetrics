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
import networkx as nx
np.set_printoptions(threshold = np.inf)
from time import time
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import texar.torch as tx
from torch.nn.utils import clip_grad_norm_

from utils.main_utils import *



parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-model", default="config_model_grade",
    help="Configuration of the model.")
parser.add_argument(
    "--config-data", default="config_data_grade", 
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
    "--output-dir", default="test_remove",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to a model checkpoint (including bert modules) to restore from.")
parser.add_argument(
    "--vis-dir", type=str, default='tensorboard',
    help="Path to save the loss and accu visulization.")
parser.add_argument(
    "--do-train", action="store_true", 
    help="Whether to run training.")
parser.add_argument(
    "--do-eval",  action="store_true", 
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--seed",
    type=int,
    required=True,
    default='71',
    help="seed for initialization")
parser.add_argument(
    "--metric-name",
    type=str,
    required=True)
parser.add_argument(
    "--dataset_dir",
    help="training dataset dir for loading graph massage",
    default='./data/DailyDialog',
    type=str)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    default='train',
    help="To rename log file.")
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed) # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_path = os.path.join('output', args.output_dir)
    tx.utils.maybe_create_dir(output_path)

    logging.root.setLevel(logging.INFO)
    logger_format_str = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    logger_format = logging.Formatter(logger_format_str)
    logger_sh = logging.StreamHandler()
    logger_sh.setFormatter(logger_format)
    logger_th = logging.FileHandler('{}_{}.log'.format(os.path.join(output_path, args.metric_name), args.task), mode='w')
    logger_th.setFormatter(logger_format)
    logging.root.addHandler(logger_sh)
    logging.root.addHandler(logger_th)


    # copy mainly file
    tx.utils.maybe_create_dir('{}/src'.format(output_path))
    os.system('cp *.py {}/src'.format(output_path))
    os.system('cp model/evaluation_model/GRADE/*.py {}/src'.format(output_path))
    os.system('cp config/*.py {}/src'.format(output_path))
    os.system('cp preprocess/*.py {}/src'.format(output_path))
    os.system('cp preprocess/utils/*.py {}/src'.format(output_path))
    os.system('cp utils/*.py {}/src'.format(output_path))
    os.system('cp *.sh {}/src'.format(output_path))

    # create vis dir
    vis_dir = os.path.join(output_path, args.vis_dir)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
    summary_writer = SummaryWriter(vis_dir)


    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # Loads data
    print("LOADING DATA....")
    config_data.train_hparam['seed'] = args.seed
    train_dataset = tx.data.MultiAlignedData(hparams=config_data.train_hparam,
                                       device=device)
    eval_dataset = tx.data.MultiAlignedData(hparams=config_data.eval_hparam,
                                      device=device)
    test_dataset = tx.data.MultiAlignedData(hparams=config_data.test_hparam,
                                      device=device)
    metric_dataset = tx.data.MultiAlignedData(hparams=config_data.metric_hparam,
                                      device=device)
    embedding_init_value = train_dataset.embedding_init_value

    # build vocab2id id2vocab
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
        {"train": train_dataset, "eval": eval_dataset, "test": test_dataset, "metric": metric_dataset}
    )
    num_train_data = config_data.num_train_data


    print("Build net....")
    model = net(args, config_model, config_data, embedding_init_value, device)
    devices_ids= [int(i) for i in args.devices_id.split(',')]
    model = nn.DataParallel(model, device_ids=devices_ids, output_device=devices_ids[0])
    model = model.to(device)
    
    left_num_train_steps = 0
    num_train_steps = int(num_train_data / config_data.train_batch_size *
                        config_data.max_train_bert_epoch)
    left_num_train_steps = num_train_steps
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)

    def _train_epoch(optim, scheduler, epoch, left_num_train_steps, eval_losses_min):
        
        logging.info("epoch: %d, learning_rate: %f", epoch, scheduler.optimizer.param_groups[0]['lr'])
        model.train()
        iterator.switch_to_dataset("train")
        mode = 'TRAINING'

        inner_step = 0
        grad_clip = 10
        for batch_id, batch in enumerate(iterator):
            avg_rec = tx.utils.AverageRecorder()
            optim.zero_grad()

            start_time = time()
            inner_step+=1

            gt_preference_label = batch["gt_preference_label"]
            pair_1_input_ids_raw_text = batch["pair_1_input_ids_raw_text"]
            pair_1_segment_ids_raw_text = batch["pair_1_segment_ids_raw_text"]
            pair_1_input_length_raw_text = (1 - (pair_1_input_ids_raw_text == 0).int()).sum(dim=1)
            pair_1_input_mask_raw_text = batch['pair_1_input_mask_raw_text']
            pair_1_input_ids_Keywords = batch["keyword_pair_1_text_ids"]
            pair_1_input_length_Keywords = batch["keyword_pair_1_length"]
            pair_1_input_ids_ctxKeywords = batch["ctx_keyword_pair_1_text_ids"]
            pair_1_input_ids_repKeywords = batch["rep_keyword_pair_1_text_ids"]
            
            pair_2_input_ids_raw_text = batch["pair_2_input_ids_raw_text"]
            pair_2_segment_ids_raw_text = batch["pair_2_segment_ids_raw_text"]
            pair_2_input_length_raw_text = (1 - (pair_2_input_ids_raw_text == 0).int()).sum(dim=1)
            pair_2_input_mask_raw_text = batch['pair_2_input_mask_raw_text']
            pair_2_input_ids_Keywords = batch["keyword_pair_2_text_ids"]
            pair_2_input_length_Keywords = batch["keyword_pair_2_length"]
            pair_2_input_ids_ctxKeywords = batch["ctx_keyword_pair_2_text_ids"]
            pair_2_input_ids_repKeywords = batch["rep_keyword_pair_2_text_ids"]

            if args.model_name == 'GRADE_K2_N10_N10' or args.model_name == 'GRADE_K2_N20_N20':
                pair_1_batched_adjs, pair_1_batch_onehop_embedding_matrix, pair_1_batch_twohop_embedding_matrix = \
                    get_adjs1(oneHop_mean_embedding_dict, twoHop_mean_embedding_dict, pair_1_input_ids_Keywords, \
                    pair_1_input_ids_ctxKeywords, pair_1_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, \
                    args.unlimit_hop)
                pair_2_batched_adjs, pair_2_batch_onehop_embedding_matrix, pair_2_batch_twohop_embedding_matrix = \
                    get_adjs1(oneHop_mean_embedding_dict, twoHop_mean_embedding_dict, pair_2_input_ids_Keywords, \
                    pair_2_input_ids_ctxKeywords, pair_2_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, \
                    args.unlimit_hop)
            else:
                pair_1_batched_adjs = get_adjs2(pair_1_input_ids_Keywords, pair_1_input_ids_ctxKeywords, \
                    pair_1_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, args.unlimit_hop)
                pair_2_batched_adjs = get_adjs2(pair_2_input_ids_Keywords, pair_2_input_ids_ctxKeywords, \
                    pair_2_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, args.unlimit_hop)
                pair_1_batch_onehop_embedding_matrix, pair_1_batch_twohop_embedding_matrix, \
                    pair_2_batch_onehop_embedding_matrix, pair_2_batch_twohop_embedding_matrix=None, None, None, None

            output_tuple = model('train', 
                    pair_1_input_ids_raw_text=pair_1_input_ids_raw_text,
                    pair_1_input_length_raw_text=pair_1_input_length_raw_text,
                    pair_1_segment_ids_raw_text=pair_1_segment_ids_raw_text,
                    pair_1_input_mask_raw_text=pair_1_input_mask_raw_text,
                    pair_1_batched_adjs=pair_1_batched_adjs,
                    pair_1_input_ids_Keywords=pair_1_input_ids_Keywords,
                    pair_1_input_length_Keywords=pair_1_input_length_Keywords,
                    pair_1_batch_onehop_embedding_matrix=pair_1_batch_onehop_embedding_matrix,
                    pair_1_batch_twohop_embedding_matrix=pair_1_batch_twohop_embedding_matrix,

                    pair_2_input_ids_raw_text=pair_2_input_ids_raw_text,
                    pair_2_input_length_raw_text=pair_2_input_length_raw_text,
                    pair_2_segment_ids_raw_text=pair_2_segment_ids_raw_text,
                    pair_2_input_mask_raw_text=pair_2_input_mask_raw_text,
                    pair_2_batched_adjs=pair_2_batched_adjs,
                    pair_2_input_ids_Keywords=pair_2_input_ids_Keywords,
                    pair_2_input_length_Keywords=pair_2_input_length_Keywords,
                    pair_2_batch_onehop_embedding_matrix=pair_2_batch_onehop_embedding_matrix,
                    pair_2_batch_twohop_embedding_matrix=pair_2_batch_twohop_embedding_matrix,
                    gt_preference_label=gt_preference_label)

            batch_size = pair_1_input_ids_raw_text.size()[0]
            avg_rec, losses, scores_tuple = add_loss_accu_msg(args, logging, avg_rec, output_tuple, batch_size)

            losses.mean().backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            scheduler.step()
            step = scheduler.last_epoch
            cur_lr = scheduler.optimizer.param_groups[0]['lr']

            end_time = time()
            batch_time = (end_time-start_time)/60 # min
            left_num_train_steps-=1
            left_time = ((left_num_train_steps)*batch_time)/60 # h

            dis_steps = config_data.display_steps
            if dis_steps > 0 and step % dis_steps == 0:
                iteration = epoch*(int(num_train_data / config_data.train_batch_size)) + inner_step
                print_loss_accu_predlabel(args, logging,  avg_rec, scores_tuple, mode, summary_writer, iteration, epoch, \
                    step, cur_lr, batch_time, left_time)

            eval_steps = config_data.eval_steps
            if eval_steps > 0 and step % eval_steps == 0:
                eval_losses_min = _eval_epoch(optim, scheduler, epoch, eval_losses_min, step) 
                _do_metrics(optim, scheduler, epoch, step)
                model.train()
            
        eval_losses_min = _eval_epoch(optim, scheduler, epoch, eval_losses_min, step)
        _do_metrics(optim, scheduler, epoch, step)

        return left_num_train_steps, eval_losses_min

    @torch.no_grad()
    def _eval_epoch(optim=None, scheduler=None, epoch=-1, eval_losses_min=None, step=-1):
        
        model.eval()
        iterator.switch_to_dataset("eval")
        mode='EVALing'

        nsamples = 0
        
        avg_rec = tx.utils.AverageRecorder()
        for batch_id, batch in enumerate(iterator):
            gt_preference_label = batch["gt_preference_label"]
            pair_1_input_ids_raw_text = batch["pair_1_input_ids_raw_text"]
            pair_1_segment_ids_raw_text = batch["pair_1_segment_ids_raw_text"]
            pair_1_input_length_raw_text = (1 - (pair_1_input_ids_raw_text == 0).int()).sum(dim=1)
            pair_1_input_mask_raw_text = batch['pair_1_input_mask_raw_text']
            pair_1_input_ids_Keywords = batch["keyword_pair_1_text_ids"]
            pair_1_input_length_Keywords = batch["keyword_pair_1_length"]
            pair_1_input_ids_ctxKeywords = batch["ctx_keyword_pair_1_text_ids"]
            pair_1_input_ids_repKeywords = batch["rep_keyword_pair_1_text_ids"]

            pair_2_input_ids_raw_text = batch["pair_2_input_ids_raw_text"]
            pair_2_segment_ids_raw_text = batch["pair_2_segment_ids_raw_text"]
            pair_2_input_length_raw_text = (1 - (pair_2_input_ids_raw_text == 0).int()).sum(dim=1)
            pair_2_input_mask_raw_text = batch['pair_2_input_mask_raw_text']
            pair_2_input_ids_Keywords = batch["keyword_pair_2_text_ids"]
            pair_2_input_length_Keywords = batch["keyword_pair_2_length"]
            pair_2_input_ids_ctxKeywords = batch["ctx_keyword_pair_2_text_ids"]
            pair_2_input_ids_repKeywords = batch["rep_keyword_pair_2_text_ids"]

            if args.model_name == 'GRADE_K2_N10_N10' or args.model_name == 'GRADE_K2_N20_N20':
                pair_1_batched_adjs, pair_1_batch_onehop_embedding_matrix, pair_1_batch_twohop_embedding_matrix = \
                    get_adjs1(oneHop_mean_embedding_dict, twoHop_mean_embedding_dict, pair_1_input_ids_Keywords, \
                    pair_1_input_ids_ctxKeywords, pair_1_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, \
                    args.unlimit_hop)
                pair_2_batched_adjs, pair_2_batch_onehop_embedding_matrix, pair_2_batch_twohop_embedding_matrix = \
                    get_adjs1(oneHop_mean_embedding_dict, twoHop_mean_embedding_dict, pair_2_input_ids_Keywords, \
                    pair_2_input_ids_ctxKeywords, pair_2_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, \
                    args.unlimit_hop)
            else:
                pair_1_batched_adjs = get_adjs2(pair_1_input_ids_Keywords, pair_1_input_ids_ctxKeywords, \
                    pair_1_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, args.unlimit_hop)
                pair_2_batched_adjs = get_adjs2(pair_2_input_ids_Keywords, pair_2_input_ids_ctxKeywords, \
                    pair_2_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, args.unlimit_hop)
                pair_1_batch_onehop_embedding_matrix=None
                pair_1_batch_twohop_embedding_matrix=None
                pair_2_batch_onehop_embedding_matrix=None
                pair_2_batch_twohop_embedding_matrix=None

            output_tuple = model('test', 
                pair_1_input_ids_raw_text=pair_1_input_ids_raw_text,
                pair_1_input_length_raw_text=pair_1_input_length_raw_text,
                pair_1_segment_ids_raw_text=pair_1_segment_ids_raw_text,
                pair_1_input_mask_raw_text=pair_1_input_mask_raw_text,
                pair_1_batched_adjs=pair_1_batched_adjs,
                pair_1_input_ids_Keywords=pair_1_input_ids_Keywords,
                pair_1_input_length_Keywords=pair_1_input_length_Keywords,
                pair_1_batch_onehop_embedding_matrix=pair_1_batch_onehop_embedding_matrix,
                pair_1_batch_twohop_embedding_matrix=pair_1_batch_twohop_embedding_matrix,

                pair_2_input_ids_raw_text=pair_2_input_ids_raw_text,
                pair_2_input_length_raw_text=pair_2_input_length_raw_text,
                pair_2_segment_ids_raw_text=pair_2_segment_ids_raw_text,
                pair_2_input_mask_raw_text=pair_2_input_mask_raw_text,
                pair_2_batched_adjs=pair_2_batched_adjs,
                pair_2_input_ids_Keywords=pair_2_input_ids_Keywords,
                pair_2_input_length_Keywords=pair_2_input_length_Keywords,
                pair_2_batch_onehop_embedding_matrix=pair_2_batch_onehop_embedding_matrix,
                pair_2_batch_twohop_embedding_matrix=pair_2_batch_twohop_embedding_matrix,
                gt_preference_label=gt_preference_label)

            batch_size = pair_1_input_ids_raw_text.size()[0]
            avg_rec, _, scores_tuple = add_loss_accu_msg(args, logging, avg_rec, output_tuple, batch_size)
            nsamples += batch_size

        print_loss_accu_predlabel(args, logging,  avg_rec, scores_tuple, mode, summary_writer, step, epoch, step)

        # save best model
        if avg_rec.avg(1) < eval_losses_min:
            eval_losses_min = avg_rec.avg(1)

            states = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(states, os.path.join('output', args.output_dir, 'model_eval_best_{}.ckpt'.format(args.seed)))
            logging.info("saving the best eval model at step: %d", step)

        return eval_losses_min

    @torch.no_grad()
    def _do_metrics(optim, scheduler, epoch, step=None):
        """Does predictions on the test set and give out the scores.
        """
        iterator.switch_to_dataset("metric")
        model.eval()
        auto_scores = []

        for batch_id, batch in tqdm(enumerate(iterator)):
            pair_1_input_ids_raw_text = batch["metric_input_ids_raw_text"]
            pair_1_segment_ids_raw_text = batch["metric_segment_ids_raw_text"]
            pair_1_input_length_raw_text = (1 - (pair_1_input_ids_raw_text == 0).int()).sum(dim=1)
            pair_1_input_mask_raw_text = batch['metric_input_mask_raw_text']
            pair_1_input_ids_Keywords = batch["keyword_pair_1_text_ids"]
            pair_1_input_length_Keywords = batch["keyword_pair_1_length"]
            pair_1_input_ids_ctxKeywords = batch["ctx_keyword_pair_1_text_ids"]
            pair_1_input_ids_repKeywords = batch["rep_keyword_pair_1_text_ids"]

            if args.model_name == 'GRADE_K2_N10_N10' or args.model_name == 'GRADE_K2_N20_N20':
                pair_1_batched_adjs, pair_1_batch_onehop_embedding_matrix, pair_1_batch_twohop_embedding_matrix = \
                    get_adjs1(oneHop_mean_embedding_dict, twoHop_mean_embedding_dict, pair_1_input_ids_Keywords, \
                    pair_1_input_ids_ctxKeywords, pair_1_input_ids_repKeywords, pair_hops, vocab2id, id2vocab, \
                    args.unlimit_hop)
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
        auto_scores = np.round(auto_scores, 4).tolist()
        
    if args.checkpoint:
        print("Loading pretrained checkpoint..........")
        ckpt = torch.load(args.checkpoint, map_location={'cuda:4':'cuda:1'})
        model.load_state_dict(ckpt['model'])

    if args.do_train:
        output_info = 'Start to training [metric_name: {}, config_model: {}, config_data: {}, model_file: {}， training_data: {}]'.format(
        args.metric_name, args.config_model, args.config_data, args.model_file, config_data.pickle_data_dir)
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        eval_losses_min = 100000
        if config_data.max_train_bert_epoch != -1:
            static_lr = 2e-5

            vars_with_decay = []
            vars_without_decay = []
            for name, param in model.named_parameters():
                if 'layer_norm' in name or name.endswith('bias'):
                    vars_without_decay.append(param)
                else:
                    vars_with_decay.append(param)

            opt_params = [{
                'params': vars_with_decay,
                'weight_decay': 0.01,
            }, {
                'params': vars_without_decay,
                'weight_decay': 0.0,
            }]
            optim = tx.core.BertAdam(
                opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr)


            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(get_lr_multiplier,
                                        total_steps=num_train_steps,
                                        warmup_steps=num_warmup_steps))

        for epoch in range(config_data.max_train_bert_epoch):
            left_num_train_steps, eval_losses_min = \
                _train_epoch(optim, scheduler, epoch, left_num_train_steps, eval_losses_min)
        
    if args.do_eval:
        logging.info("=============Start to eval=============")
        _eval_epoch(epoch=0)

    summary_writer.close() 


if __name__ == "__main__":
    main()