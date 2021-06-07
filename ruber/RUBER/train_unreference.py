#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.3.28


'''
This file is the model for automatic dialog evaluation
refer to:
RUBER: An Unsupervised Method for Automatic Evaluation of Open-Domain Dialog Systems
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import argparse
import random
import os
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm

from unreference_score import *
from utils import *

# set the random seed for the model
random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
    
    
parser = argparse.ArgumentParser(description='EPN-RUBER utils script')
parser.add_argument('--seed', type=int, default=123, 
                    help='seed for random init')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, 
                    help='weight decay ratio')
parser.add_argument('--dataset', type=str, default='xiaohuangji', 
                    help='the dataset we used')

args = parser.parse_args()


def train(data_iter, net, optimizer, 
          delta=0.5, grad_clip=10):
    net.train()
    batch_num, losses = 0, 0
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, qlength, rlength, label = batch
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        qlength = torch.from_numpy(qlength)
        rlength = torch.from_numpy(rlength)
        label = torch.from_numpy(label).float()
        batch_size = len(qlength)
                
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            qlength, rlength = qlength.cuda(), rlength.cuda()
            label = label.cuda()
            
            qbatch = qbatch.transpose(0, 1)
            rbatch = rbatch.transpose(0, 1)
        
        optimizer.zero_grad()
        
        scores = net(qbatch, qlength, rbatch, rlength)    # [2 * B]
        # pos_scores, neg_scores = torch.split(scores, [int(batch_size / 2), 
        #                                               int(batch_size / 2)])
        # loss = criterion(pos_scores, 
        #                  neg_scores, 
        #                  torch.ones(pos_scores.shape[0]).cuda())
        #import ipdb; ipdb.set_trace()
        loss = criterion(scores, label)
        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        
        optimizer.step()
        losses += loss.item()
        batch_num = batch_idx + 1
        
    return round(losses / batch_num, 4)


# validation
def validation(data_iter, net, delta=0.8):
    ''' 
    calculate the Acc
    '''
    net.eval()
    losses, batch_num, acc, acc_num = 0, 0, 0, 0
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    for batch_idx, batch in enumerate(data_iter):
        qbatch, rbatch, qlength, rlength, label = batch
        qbatch = torch.from_numpy(qbatch)
        rbatch = torch.from_numpy(rbatch)
        qlength = torch.from_numpy(qlength)
        rlength = torch.from_numpy(rlength)
        label = torch.from_numpy(label).float()
        batch_size = len(qlength)
                
        if torch.cuda.is_available():
            qbatch, rbatch = qbatch.cuda(), rbatch.cuda()
            qlength, rlength = qlength.cuda(), rlength.cuda()
            label = label.cuda()
            
            qbatch = qbatch.transpose(0, 1)
            rbatch = rbatch.transpose(0, 1)
                    
        scores = net(qbatch, qlength, rbatch, rlength)    # [2 * B]

        loss = criterion(scores, label)
        
        # Cross Entropy
        # s = torch.argmax(scores.log_softmax(dim=1), dim=1)
        # acc += torch.sum(s == label).item()
        
        # BCELoss
        s = scores >= 0.5
        acc += torch.sum(s.float() == label).item()
        
        acc_num += batch_size
        
        batch_num += 1
        losses += loss.item()
    
    return round(losses / (batch_num), 4), round(acc / acc_num, 4)


def test(net, test_data):
    # ipdb.set_trace()
    test_loss, test_acc = validation(test_data, net)
    
    # add the test for Per Dialogue metric
    print('[!] test_loss:', test_loss)
    print('[!] test_Acc', test_acc)
    
    
def main(trainqpath, trainrpath, devqpath, devrpath,
         testqpath, testrpath, weight_decay=1e-4, lr=1e-3):
    # if training acc is nearly 0.5, try to use lr = 1e-4, 
    # because of the unstable training environment
    with open(f'data/{args.dataset}/src-vocab.pkl', 'rb') as f:
        srcv = pickle.load(f)
        
    with open(f'data/{args.dataset}/tgt-vocab.pkl', 'rb') as f:
        tgtv = pickle.load(f)
    
    net = RUBER_unrefer(srcv.get_vocab_size(), tgtv.get_vocab_size(),
                        100, 100, SOURCE=srcv, TARGET=tgtv)
    if torch.cuda.is_available():
        net.cuda()
    print('[!] Finish init the vocab and net')
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    epoches = 100
    grad_clip = 10
    early_stop_patience = 10
    pbar = tqdm(range(1, epoches + 1))
    training_losses, validation_losses, validation_metrices = [], [], []
    min_loss = np.inf
    best_metric = -1
    
    # clear the result
    os.system(f"rm ./ckpt/{args.dataset}/*")
    print(f"[!] Clear the checkpoints under ckpt")
    
    patience = 0
    begin_time = time.time()
    idxx = 1
    for epoch in pbar:
        train_iter = get_batch(trainqpath, trainrpath, 128, seed=args.seed)
        dev_iter = get_batch(devqpath, devrpath, 128, args.seed)
        test_iter = get_batch(testqpath, testrpath, 128, args.seed)
    
        training_loss = train(train_iter, net, optimizer)
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
            #torch.save(state,
            #f'./ckpt/{args.dataset}/Acc_{validation_metric}_vloss_{validation_loss}_epoch_{epoch}.pt')        
            torch.save(state,
            f'./ckpt/{args.dataset}/best_ckpt.pt')        
        else:
            patience += 1
            
        # Save the model

        pbar.set_description(f"loss(train-dev): {training_loss}-{validation_loss}, Acc: {validation_metric}, patience: {patience}")
        idxx += 1
        
        if patience > early_stop_patience:
            break
            print('[!] early stop')
    
    pbar.close()

    # calculate costing time
    end_time = time.time()
    hour = math.floor((end_time - begin_time) / 3600)
    minute = math.floor(((end_time - begin_time) - 3600 * hour) / 60)
    second = (end_time - begin_time) - hour * 3600 - minute * 60
    print(f"Cost {hour}h, {minute}m, {round(second, 2)}s")
    
    # load best and test
    load_best_model(args.dataset, net)
    
    # test
    test(net, test_iter)
    

if __name__ == "__main__":
    # train the model
    main(f'./data/{args.dataset}/src-train-id.pkl',
         f'./data/{args.dataset}/tgt-train-id.pkl',
         f'./data/{args.dataset}/src-dev-id.pkl',
         f'./data/{args.dataset}/tgt-dev-id.pkl',
         f'./data/{args.dataset}/src-test-id.pkl',
         f'./data/{args.dataset}/tgt-test-id.pkl',
         lr=args.lr, weight_decay=args.weight_decay)
