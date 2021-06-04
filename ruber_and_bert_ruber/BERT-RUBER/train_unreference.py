#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.7.10

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import ipdb
import sys

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
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel

# set the random seed for the model
random.seed(20)
torch.manual_seed(20)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(20)
    
    
def train(data_iter, net, optimizer, grad_clip=10):
    net.train()
    batch_num, losses = 0, 0
    criterion = nn.BCELoss()
    #criterion = nn.BCEWithLogitsLoss()
    
    for batch_idx, batch in tqdm(enumerate(data_iter)):
        #qbatch = torch.from_numpy(qbatch)
        #rbatch = torch.from_numpy(rbatch)
        #label = torch.from_numpy(label).float()
        #batch_size = qbatch.shape[0]

        label = batch['labels']
        batch_size = label.shape[0]
        label = label.view(-1).to('cuda')

        optimizer.zero_grad()
        
        linear, scores = net(batch)
        loss = criterion(scores, label)
        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        
        if batch_idx % 200 == 0:
            print(loss.item())
        losses += loss.item()
        batch_num = batch_idx + 1

    return round(losses / batch_num, 4)

def validation(data_iter, net):
    net.eval()

    losses, batch_num, acc, acc_num = 0, 0, 0, 0
    criterion = nn.BCELoss()
    
    for batch_idx, batch in tqdm(enumerate(data_iter)):
        #qbatch, rbatch, label = batch 
        #qbatch = torch.from_numpy(qbatch)
        #rbatch = torch.from_numpy(rbatch)
        #label = torch.from_numpy(label).float()
        #batch_size = qbatch.shape[0]
        label = batch['labels']
        batch_size = label.shape[0]
        label = label.view(batch_size * 2).to('cuda')
    
        
        with torch.no_grad():
            linear, scores = net(batch)
            loss = criterion(scores, label)
        
        s = scores >= 0.5
        acc += torch.sum(s.float() == label).item()
        acc_num += len(label)
        
        batch_num += 1
        losses += loss.item()
        
    return round(losses / batch_num, 4), round(acc / acc_num, 4)


def test(net, data_iter):
    test_loss, test_acc = validation(data_iter, net)
    
    print('[!] test_loss:', test_loss)
    print('[!] test_Acc', test_acc)

def prepare_data(tokenizer, qpath, rpath):
    with open(qpath, 'rb') as f:
        qdataset = pickle.load(f)
    
    qdataset = [' '.join(q.split()[-128:]) for q in qdataset]
    print(qdataset[:5])

    with open(rpath, 'rb') as f:
        rdataset = pickle.load(f)
    print(rdataset[:5])
    print('Num samples', len(qdataset))


    q_encoding = tokenizer(qdataset, padding=True, truncation=True)
    r_encoding = tokenizer(rdataset, padding=True, truncation=True)
    return q_encoding, r_encoding

class Dataset(torch.utils.data.Dataset):
    def __init__(self, q_encodings, r_encodings):
        self.q_encodings = q_encodings
        self.r_encodings = r_encodings
        self.data_num = len(q_encodings['input_ids'])
        self.neg_idx = np.random.choice(self.data_num, self.data_num)
    
    def __getitem__(self, idx):
        item = {}
        for key, val in self.q_encodings.items():
            tensor = torch.tensor(val[idx])
            item[f'q_{key}'] = torch.stack([tensor, tensor])

        for key, val in self.r_encodings.items():
            p_tensor, n_tensor = torch.tensor(val[idx]), torch.tensor(val[self.neg_idx[idx]])
            item[f'r_{key}'] = torch.stack([p_tensor, n_tensor])
        item['labels'] = torch.tensor([1, 0], dtype=torch.float32)
        return item
    
    def __len__(self):
        return self.data_num

def create_dataloader(q_encoding, r_encoding, b_size):
    dataset = Dataset(q_encoding, r_encoding)
    return torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=True)

def main(trainqpath, trainrpath, devqpath, devrpath, testqpath, testrpath, dataset,
         weight_decay=1e-4, lr=1e-3):
    net = BERT_RUBER_unrefer(768, dropout=0.5)
    if torch.cuda.is_available():
        net.cuda()
    print('[!] Finish init the model')
    
    optimizer = optim.Adam(net.parameters(), lr=3e-5, weight_decay=weight_decay)
    #optimizer = AdamW(net.parameters(), lr=3e-5)
 
    #epoches, grad_clip, b_size = 200, 10, 64
    epoches, grad_clip, b_size = 100, 10, 64
    #epoches, grad_clip, b_size = 5, 1, 8
    pbar = tqdm(range(1, epoches + 1))
    training_losses, validation_losses, validation_metrices = [], [], []
    min_loss = np.inf
    best_metric = -1
    
    os.system(f'rm ./ckpt/{dataset}/*')
    print(f'[!] Clear the checkpoints under ckpt')
    
    patience = 0
    begin_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    q_train, r_train = prepare_data(tokenizer, trainqpath, trainrpath)
    q_dev, r_dev = prepare_data(tokenizer, devqpath, devrpath)
    q_test, r_test = prepare_data(tokenizer, testqpath, testrpath)

    dev_iter = create_dataloader(q_dev, r_dev, b_size)
    test_iter = create_dataloader(q_test, r_test, b_size)

    for epoch in pbar:
        
        
        train_iter = create_dataloader(q_train, r_train, b_size)
        training_loss = train(train_iter, net, optimizer, grad_clip)
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
                f'./ckpt/{dataset}/best_model.pt')
                #f'./ckpt/{dataset}/Acc_{validation_metric}_vloss_{validation_loss}_epoch_{epoch}.pt')
        else:
            patience += 1
            
      
        if patience > 20:
            print(f'[!] early stop')
            #break
        
        pbar.set_description(f"loss(train-dev): {training_loss}-{validation_loss}, Acc: {validation_metric}, patience: {patience}")
    pbar.close()
    
    # calculate costing time
    end_time = time.time()
    hour = math.floor((end_time - begin_time) / 3600)
    minute = math.floor(((end_time - begin_time) - 3600 * hour) / 60)
    second = (end_time - begin_time) - hour * 3600 - minute * 60
    print(f"Cost {hour}h, {minute}m, {round(second, 2)}s")
    
    # load best and test
    # load_best_model(net)
    
    # test
    # test(net, test_iter)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--dataset', type=str, default=None, help='')
    
    args = parser.parse_args()

    # show the parameters
    print('[!] Parameters:')
    print(args)

    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
    
        main(f'data/{args.dataset}/src-train.embed',
        f'data/{args.dataset}/tgt-train.embed',
        f'data/{args.dataset}/src-dev.embed',
        f'data/{args.dataset}/tgt-dev.embed',
        f'data/{args.dataset}/src-test.embed',
        f'data/{args.dataset}/tgt-test.embed',
        args.dataset)
