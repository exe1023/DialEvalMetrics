#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.7.10

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np

import argparse
import random
import os
import time
import pprint
import ipdb
import math
from tqdm import tqdm

from unreference_score import *
from utils import *


parser = argparse.ArgumentParser(description='EPN-RUBER training script')
parser.add_argument('--temp', type=float, default=0.1,
                    help='temperature coefficient for weighted negative sampling')
parser.add_argument('--src_train', type=str, default='data/xiaohuangji/src-train.embed',
                    help='source training dataset')
parser.add_argument('--tgt_train', type=str, default='data/xiaohuangji/tgt-train.embed',
                    help='target training dataset')
parser.add_argument('--fluent_path', type=str, default='', help='')
parser.add_argument('--safe_path', type=str, default='', help='')
parser.add_argument('--src_dev', type=str, default='data/xiaohuangji/src-dev.embed',
                    help='source validation dataset')
parser.add_argument('--tgt_dev', type=str, default='data/xiaohuangji/tgt-dev.embed',
                    help='target validation dataset')
parser.add_argument('--src_test', type=str, default='data/xiaohuangji/src-test.embed',
                    help='source test dataset')
parser.add_argument('--tgt_test', type=str, default='data/xiaohuangji/tgt-test.embed',
                    help='target test dataset')
parser.add_argument('--pre_choice', type=int, default=500,
                    help='previously sample some batches for generating negative batch')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay to address overfitting')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--grad_clip', type=float, default=10,
                    help='gradient clip')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--epoches', type=int, default=100,
                    help='the number of iterations')
parser.add_argument('--model_name', type=str, default='big-no-weight',
                    help='model name of different versions')
parser.add_argument('--step', type=int, default=5,
                    help='the step that exchange the parameters to target network')
parser.add_argument('--da_src_train', type=str, default='./data/xiaohuangji/src-train-da5.embed',
                    help='data augmentation source training dataset')
parser.add_argument('--da_tgt_train', type=str, default='./data/xiaohuangji/tgt-train-da5.embed',
                    help='data augmentation target training dataset')
parser.add_argument('--pretrained_model', type=str, default='',
                    help='the pretrained model dir')
parser.add_argument('--bert_size', type=int, default=768,
                    help='the bert embedding size (3072 for REDUCE_MEAN / REDUCE_MAX or 1536 for REDUCE_MAX_MEAN)')
parser.add_argument('--weight_step', type=int, default=5,
                    help='the weighted step for batch generator')
parser.add_argument('--epoch_threshold', type=int, default=30,
                    help='the threshold of epoch for loading best performance model')
parser.add_argument('--weight_matrix', type=str, default='./data/xiaohuangji/bert-weight-matrix.pkl',
                    help='the weight matrix, default for bert')
parser.add_argument('--dataset', type=str, default='xiaohuangji',
                    help='the dataset we used')
parser.add_argument('--fuzzy_threshold', type=float, default=0.2,
                    help='Value range (0, 0.5]. The fuzzy threshold for filting the wrong label dataset in the augmentation data, which can update the model robustly. 50% threshold will make no filting for the data augmentation.')
parser.add_argument('--fuzzy_process', type=str, default='drop',
                    help='the way to process the fuzzy augmentation data, drop or change, drop will throw away all the fuzzy data and change will set all the fuzzy data as the positive to')
parser.add_argument('--enhance', dest='enhance', action='store_true')
parser.add_argument('--no-enhance', dest='enhance', action='store_false')
parser.add_argument('--weight', dest='weight', action='store_true')
parser.add_argument('--no-weight', dest='weight', action='store_false')
parser.add_argument('--da', dest='da', action='store_true')
parser.add_argument('--no-da', dest='da', action='store_false')
parser.add_argument('--bm25', dest='bm25', action='store_true')
parser.add_argument('--no-bm25', dest='bm25', action='store_false')
parser.add_argument('--da_times', type=int, default=5,
                    help='the times of the augmentation data')
parser.add_argument('--seed', type=int, default=123, 
                    help='seed for random init')
parser.add_argument('--patience', type=int, default=20, 
                    help='patience for training early stop')

args = parser.parse_args()

# set the random seed for the model
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    
    
def train(data_iter, net, optimizer, grad_clip=10):
    net.train()
    batch_num, losses = 0, 0
    criterion = nn.BCELoss()
    
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, label = batch
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        label = torch.from_numpy(label).float()
        batch_size = qbatch.shape[0]
        
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            label = label.cuda()
            
        optimizer.zero_grad()
        
        scores = net(qbatch, rbatch)         # [B, 3]
        loss = criterion(scores, label)
        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        
        losses += loss.item()
        batch_num = batch_idx + 1
    return round(losses / batch_num, 4)


def validation(data_iter, net):
    net.eval()
    losses, batch_num, acc, acc_num = 0, 0, 0, 0
    criterion = nn.BCELoss()
    
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, label = batch 
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        label = torch.from_numpy(label).float()
        batch_size = qbatch.shape[0]
                
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            label = label.cuda()
            
        scores = net(qbatch, rbatch)
        loss = criterion(scores, label)
        
        s = scores >= 0.5
        acc += torch.sum(s.float() == label).item()
        acc_num += batch_size
        
        batch_num += 1
        losses += loss.item()
        
    return round(losses / batch_num, 4), round(acc / acc_num, 4)


def test(net, data_iter):
    test_loss, test_acc = validation(data_iter, net)
    
    print('[!] test_loss:', test_loss)
    print('[!] test_Acc', test_acc)
    
    
def main(trainqpath, trainrpath, devqpath, devrpath, testqpath, testrpath, 
         weight_decay=1e-4, lr=1e-3, t=0.01, weighted=False, prechoice=200,
         grad_clip=10, epoches=100, batch_size=256, step=5, enhance=False,
         fluent_path=None, safe_path=None):
    '''enhance model noted:
    enhance model need to use the pretrained model,
    donot train the target network
    '''
    # init the network
    net = BERT_RUBER_unrefer(args.bert_size, dropout=0.5)
    if enhance:
        target_net = BERT_RUBER_unrefer(args.bert_size, dropout=0.5)
        load_best_model(args.dataset, 
                        args.pretrained_model, target_net, threshold=args.epoch_threshold)
        load_best_model(args.dataset, 
                        args.pretrained_model, net, threshold=args.epoch_threshold)
    if torch.cuda.is_available():
        net.cuda()
        if enhance:
            target_net.cuda()
    print('[!] Finish init the model')
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    pbar = tqdm(range(1, epoches + 1))
    training_losses, validation_losses, validation_metrices = [], [], []
    min_loss = np.inf
    best_metric = -1
    
    patience = 0
    begin_time = time.time()
    
    # ipdb.set_trace()
    if weighted and not args.bm25:
        weight_matrix = read_file(args.weight_matrix)
        print('[!] Load the weight matrix over')
    else:
        weight_matrix = None
    
    for epoch in pbar:
        # only enhance the training dataset
        # random sample in weight mode
        random_weight = epoch % args.weight_step == 0
        if enhance:
            if epoch % step == 1:
                # update target network
                target_net.load_state_dict(net.state_dict())
                dalabels = get_da_label(target_net, args.da_src_train, 
                                        args.da_tgt_train, batch_size, 
                                        fuzzy_threshold=args.fuzzy_threshold)
            try:
                if weighted:
                    train_iter = get_batch_da(dalabels, trainqpath, trainrpath,
                                          args.da_src_train, args.da_tgt_train,
                                          batch_size, weighted=weighted, 
                                          prechoice=prechoice, t=t,
                                          weight_matrix=weight_matrix,
                                          weight_step=random_weight,
                                          seed=args.seed,
                                          fuzzy_threshold=args.fuzzy_threshold,
                                          fuzzy_process=args.fuzzy_process,
                                          fluent_path=fluent_path, dataset=args.dataset,
                                          bm25=args.bm25)
                else:
                    train_iter = get_batch_da(dalabels, trainqpath, trainrpath,
                                          args.da_src_train, args.da_tgt_train,
                                          batch_size, weighted=weighted, 
                                          prechoice=prechoice, t=t,
                                          weight_step=random_weight,
                                          seed=args.seed,
                                          fuzzy_threshold=args.fuzzy_threshold,
                                          fuzzy_process=args.fuzzy_process,
                                          fluent_path=fluent_path, dataset=args.dataset,
                                          bm25=args.bm25)
            except:
                raise Exception('Enhance dataset load failed !')
        else:
            if weighted:
                train_iter = get_batch(trainqpath, trainrpath, batch_size, 
                                       weighted=weighted, prechoice=prechoice, t=t,
                                       weight_matrix=weight_matrix,
                                       weight_step=random_weight,
                                       seed=args.seed,
                                       da=args.da,
                                       times=args.da_times,
                                       fluent_path=fluent_path,
                                       safe_path=safe_path,
                                       bm25=args.bm25,
                                       dataset=args.dataset)
            else:
                train_iter = get_batch(trainqpath, trainrpath, batch_size, 
                                       weighted=weighted, prechoice=prechoice, t=t,
                                       weight_step=random_weight,
                                       seed=args.seed,
                                       da=args.da,
                                       times=args.da_times,
                                       fluent_path=fluent_path,
                                       safe_path=safe_path,
                                       bm25=args.bm25,
                                       dataset=args.dataset)
        
        # dev and test mode do not need the weighted, it's just waste the time
        dev_iter = get_batch(devqpath, devrpath, batch_size, 
                             weighted=False, prechoice=prechoice, 
                             bm25=args.bm25,
                             t=t, seed=args.seed, fluent_path=None, dataset=args.dataset)
        test_iter = get_batch(testqpath, testrpath, batch_size, 
                             weighted=False, prechoice=prechoice, 
                             bm25=args.bm25,
                             t=t, seed=args.seed, fluent_path=None, dataset=args.dataset)
        
        # get the batch over
        
        training_loss = train(train_iter, net, optimizer, grad_clip=grad_clip)
        validation_loss, validation_metric = validation(dev_iter, net)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        validation_metrices.append(validation_metric)
        
        if best_metric < validation_metric:
            patience = 0
            best_metric = validation_metric
            min_loss = validation_loss
            state = {'net': net.state_dict(), 
                 'optimizer': optimizer.state_dict(), 
                 'epoch': epoch}
            torch.save(state,
                f'./ckpt/{args.dataset}/{args.model_name}/best_model.ckpt')     
                #f'./ckpt/{args.dataset}/{args.model_name}/Acc_{validation_metric}_vloss_{validation_loss}_epoch_{epoch}_weighted_{weighted}.pt')     
        else:
            patience += 1
            
        if patience > args.patience:
            print(f'[!] Early stop at epoch {epoch}')
            break
            
      
        pbar.set_description(f"loss(train-dev): {training_loss}-{validation_loss}, Acc: {validation_metric}, patience: {patience}")
        
    pbar.close()
    
    # calculate costing time
    end_time = time.time()
    hour = math.floor((end_time - begin_time) / 3600)
    minute = math.floor(((end_time - begin_time) - 3600 * hour) / 60)
    second = (end_time - begin_time) - hour * 3600 - minute * 60
    print(f"Cost {hour}h, {minute}m, {round(second, 2)}s")
    
    # load best and test
    load_best_model(args.dataset, args.model_name, net, threshold=args.epoch_threshold)
    
    # test
    test(net, test_iter)
    


if __name__ == "__main__":
    print('Parameters: ', end='')
    pprint.pprint(args)
    main(args.src_train,
         args.tgt_train,
         args.src_dev,
         args.tgt_dev,
         args.src_test,
         args.tgt_test,
         weighted=args.weight,
         t=args.temp,
         weight_decay=args.weight_decay,
         lr=args.lr,
         grad_clip=args.grad_clip,
         prechoice=args.pre_choice,
         epoches=args.epoches,
         batch_size=args.batch_size,
         step=args.step,
         enhance=args.enhance,
         fluent_path=args.fluent_path,
         safe_path=args.safe_path)
