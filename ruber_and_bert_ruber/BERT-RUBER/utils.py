#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.7.10

'''
utils file contains the tool function
1. bert embedding collect (query, groundtruth, generated)
2. load_best_model
3. batch iterator
'''

import pickle
import bz2
import numpy as np
import os
import re
#from bert_serving.client import BertClient
import argparse
from transformers import AutoTokenizer, AutoModel


def load_best_model(net, dataset):
    import torch
    path = f"./ckpt/{dataset}/"
    best_acc, best_file = -1, None
    best_epoch = -1
    
    for file in os.listdir(path):
        try:
            _, acc, _, loss, _, epoch = file.split("_")
            epoch = epoch.split('.')[0]
        except:
            continue
        acc = float(acc)
        epoch = int(epoch)
        if epoch > best_epoch:
        # if acc > best_acc:
            best_file = file
            best_epoch = epoch
            # best_acc = acc
    
    best_file = 'best_model.pt'
    if best_file:
        file_path = path + best_file
        print(f'[!] Load the model from {file_path}')
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception(f"[!] No saved model")


def get_batch(qpath, rpath, batch_size):
    # bert embedding matrix, [dataset_size, 768]
    # return batch shape: [B, 768]

    size = len(qdataset)
    idx = 0
    print(f'Total: {size}, Batch: {batch_size}, Estimated Batch Num: {size // batch_size}')
    while True:
        qbatch = qdataset[idx:idx+batch_size]
        rbatch = rdataset[idx:idx+batch_size]
        #pidx = np.random.choice(rdataset.shape[0], batch_size)
        pidx = np.random.choice(len(rdataset), len(rbatch))
        
        #nbatch = rdataset[pidx]
        for idx in pidx:
            rbatch.append(rdataset[idx])
        
        #qbatch = np.concatenate([qbatch, qbatch])
        #rbatch = np.concatenate([rbatch, nbatch])
        qbatch = qbatch * 2
        
        #label = np.concatenate([np.ones(int(qbatch.shape[0] / 2)),
        #                        np.zeros(int(qbatch.shape[0] / 2))])
        label = np.concatenate([np.ones(len(qbatch) // 2), np.zeros(len(qbatch) // 2)])
        
        # shuffle
        #pureidx = np.arange(qbatch.shape[0])
        #pureidx = np.arange(len(qbatch))
        #np.random.shuffle(pureidx)
        #qbatch = qbatch[pureidx]
        #rbatch = rbatch[pureidx]
        #label = label[pureidx]
        
        idx += batch_size
        yield qbatch, rbatch, label
        
        if idx > size:
            break
    return None
    

def process_train_file(path, embed_type, batch_size=128):
    # batch_size: batch for bert to feedforward
    #bc = BertClient()
    contexts, responses = [], []
    # non-multi-turn
    with open(path) as f:
        for line in f.readlines():
            #dataset.append(' '.join(line.strip().split()[-200:]))
            data = line.strip().split('__eou__')
            data = data[:-1] # remove the last one
            for idx in range(1, len(data) - 1):
                context = ' '.join(data[:idx])
                response = data[idx]
                contexts.append(context)
                responses.append(response)
    # multi-turn
    # with open(path) as f:
    #     for line in f.readlines():
    #         dataset.append(line.strip().split('__eou__')[-100:])
    
    # bert-as-serive
    # embed = []
    # idx = 0
    # from itertools import accumulate
    # while True:
    #     nbatch = dataset[idx:idx+batch_size]
    #     batch = []
    #     for i in nbatch:
    #         batch += i
    #     batch_length = list(accumulate([len(i) for i in nbatch]))
    #     batch_length = [0] + batch_length
    #     rest = bc.encode(batch)
    #     fr = []
    #     for i in range(1, len(batch_length)):
    #         fr.append(np.sum(rest[batch_length[i-1]:batch_length[i]], axis=0))
    #     embed.append(np.stack(fr))    # [b, 768]
    #     idx += batch_size
    #     if idx > len(dataset):
    #         break
    #     print(f'{path}: {idx} / {len(dataset)}', end='\r')
    # embed = np.concatenate(embed)
    # print(f'embed shape: {embed.shape}')
    
    # no-multi-turn
    """
    embed = []
    idx = 0
    while True:
        batch = dataset[idx:idx+batch_size]
        #rest = bc.encode(batch)    # [batch_size, 768]
        embed.append(rest)
        idx += batch_size
        if idx > len(dataset):
            break
        print(f'{idx} / {len(dataset)}', end='\r')
    #embed = np.concatenate(embed)  # [dataset_size, 768]
    # [dataset_size]
    """
    
    with open(f'data/dailydialog/src-{embed_type}.embed', 'wb') as f:
        pickle.dump(contexts, f)
    with open(f'data/dailydialog/tgt-{embed_type}.embed', 'wb') as f:
        pickle.dump(responses, f)
        
    #print(f'Write the bert embedding into {embed_path}')
    
    
def cal_avf_performance(path):
    su, sr, u = [], [], []
    with open(path) as f:
        p = re.compile('(0\.[0-9]+)\((.+?)\)')
        for line in f.readlines():
            m = p.findall(line.strip())
            if 'su_p' in line:
                su.append(m)
            elif 'sr_p' in line:
                sr.append(m)
            elif 'u_p' in line:
                u.append(m)
            else:
                raise Exception("Wrong file format !")
    # cal avg performance
    avg_u_p, avg_u_s, avg_ruber_p, avg_ruber_s = [], [], [], []
    for ku, ru in zip(su, u):
        avg_u_p.append(float(ku[0][0]))
        avg_u_s.append(float(ku[1][0]))
        avg_ruber_p.append(float(ru[0][0]))
        avg_ruber_s.append(float(ru[1][0]))
    print(f'Unrefer Avg pearson: {round(np.mean(avg_u_p), 5)}, Unrefer Avg spearman: {round(np.mean(avg_u_s), 5)}')
    print(f'RUBER Avg pearson: {round(np.mean(avg_ruber_p), 5)}, RUBER Avg spearman: {round(np.mean(avg_ruber_s), 5)}')
        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()
    if args.mode == 'calculate':
        cal_avf_performance(f'./data/{args.dataset}/result.txt')
    elif args.mode == 'process':
        process_train_file(f'./data/{args.dataset}/train/dialogues_train.txt', 'train')
        #process_train_file(f'./data/{args.dataset}/tgt-train.txt', 
        #                   f'./data/{args.dataset}/tgt-train.embed')
        process_train_file(f'./data/{args.dataset}/validation/dialogues_validation.txt', 'dev')
        #process_train_file(f'./data/{args.dataset}/tgt-dev.txt', 
        #                   f'./data/{args.dataset}/tgt-dev.embed')
        process_train_file(f'./data/{args.dataset}/test/dialogues_test.txt', 'test')
        #process_train_file(f'./data/{args.dataset}/tgt-test.txt', 
        #                   f'./data/{args.dataset}/tgt-test.embed')
