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
import os
import sys
import re
import time
import csv
import math
import random
import pickle


def tokenizer(iterator):
    for value in iterator:
        yield value.split()
        

def load_embedding(VOCAB, path, embedding_dim=300):
    with open(path) as f:
        weights = np.random.rand(VOCAB.get_vocab_size(), embedding_dim)
        counter = 0
        for line in f.readlines():
            try:
                line = line.strip().split()
                v = list(map(float, line[1:]))
                word = line[0]
                wid = VOCAB.get_index(word)
                if wid != VOCAB.get_index("<unk>"):
                    counter += 1
                    weights[wid] = np.array(v)
            except Exception as e:
                print(e)
                ipdb.set_trace()
        print(f"[!] Loading the weights {round(counter / VOCAB.get_vocab_size(), 4)}")
    return weights
def get_best_model(dataset, net):
    file_path = f"./ckpt/{dataset}/best_ckpt.pt"
    print(f'[!] Load the model from {file_path}')
    net.load_state_dict(torch.load(file_path)['net']) 

    
def load_best_model(dataset, net):
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
        # if epoch > best_epoch:
        if acc > best_acc:
            best_file = file
            # best_epoch = epoch
            best_acc = acc

    if best_file:
        file_path = path + best_file
        print(f'[!] Load the model from {file_path}')
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception(f"[!] No saved model")
        
        
def load_special_model(net, path):
    try:
        net.load_state_dict(torch.load(path)['net'])
    except:
        raise Exception(f"[!] {path} load error")
        
        
class Vocab():
    
    '''
    The vocab instance for the dataset (volumn)
    '''
    
    def __init__(self, special_tokens, lower=True):
        self.special_tokens = special_tokens.copy()
        self.freq = {}
        self.lower = lower
     
    def add_token(self, token):
        token = token.lower()
        if token not in self.freq:
            self.freq[token] = 1
        else:
            self.freq[token] += 1
            
    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)
            
    def get_vocab_size(self):
        try:
            return len(self.itos)
        except:
            raise Exception("Not init, try to call .squeeze func")
            
    def get_index(self, token):
        if token in self.stoi:
            return self.stoi[token]
        else:
            return self.stoi["<unk>"]
            
    def get_token(self, index):
        if index > len(self.itos):
            raise Exception("Bigger than vocab size")
        else:
            return self.itos[index]
        
    def squeeze(self, threshold=1, max_size=None, debug=False):
        '''
        threshold for cutoff
        max_size for constraint the size of the vocab
        threshold first, max_size next
        
        this function must be called to get tokens list
        '''
        words = list(self.freq.items())
        words.sort(key=lambda x: x[1])
        new_words = []
        for word in words:
            if word[1] >= threshold:
                new_words.append(word)
        words = list(reversed(new_words))
        if max_size and len(words) > max_size:
            words = words[:max_size]
        self.itos = [word for word, freq in words]
        
        # add the special tokens
        if self.special_tokens:
            self.itos.extend(self.special_tokens)
        
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}
        
        del self.freq
        if debug:
            print(f"Vocab size: {len(self.stoi)}")
            
    
def process_train_file(path, vocabpath, idpath, max_length=128, mode='train'):
    # create the vocab instance for models
    # create the id file for training dataset
    if mode == 'train':
        vocab = Vocab(['<unk>', "<pad>", "<sos>", "<eos>"])
        with open(path) as f:
            for line in f.readlines():
                words = line.strip().split()
                vocab.add_tokens(words)
        vocab.squeeze(max_size=20000, debug=True)
        
        # save the vocab into file
        with open(vocabpath, 'wb') as f:
            pickle.dump(vocab, f)
    else:
        with open(vocabpath, 'rb') as f:
            vocab = pickle.load(f)
        
    # creat the dataset
    with open(path) as f:
        dataset = []
        ll = []
        for line in f.readlines():
            words = line.strip().split()[:max_length]
            words = ["<sos>"] + words + ["<eos>"]
            length = len(words)
            if len(words) < max_length + 2:
                words.extend(['<pad>'] * (max_length + 2 - len(words)))
            dataset.append(np.array([vocab.get_index(word) for word in words]))
            ll.append(length)
        dataset = np.stack(dataset)    # [B, Max_length]
        length  = np.array(ll)
    
    # save the id training dataset into the file
    with open(idpath, 'wb') as f:
        pickle.dump((length, dataset), f)
        
        
def make_embedding_matrix(fname, word2vec, vec_dim, fvocab):
    if os.path.exists(fname):
        print('Loading embedding matrix from %s'%fname)
        return pickle.load(open(fname, 'rb'))

    with open(fvocab, 'rb') as f:
        vocab = pickle.load(f)
    print('Saving embedding matrix in %s'%fname)
    matrix={}
    counter = 0
    for v in vocab.stoi:
        vec = word2vec[v] if v in word2vec \
                else [0.0 for _ in range(vec_dim)]
        if v in word2vec:
            counter += 1
        matrix[v] = vec
    pickle.dump(matrix, open(fname, 'wb'), protocol=2)
    print(f'Covery: {counter} / {vocab.get_vocab_size()}')
    return matrix


def load_word2vec(fword2vec):
    """
    Return:
        word2vec dict
        vector dimension
        dict size
    """
    print('Loading word2vec dict from %s'%fword2vec)
    vecs = {}
    vec_dim=0
    with open(fword2vec) as fin:
        # size, vec_dim = list(map(int, fin.readline().split()))
        vec_dim = 300
        size = 0
        for line in fin:
            ps = line.rstrip().split()
            try:
                vecs[ps[0]] = list(map(float, ps[1:]))
                size += 1
            except:
                pass
    return vecs, vec_dim, size


def get_batch(qpath, rpath, batch_size, seed=100):
    # getting batch for training unreference score
    # yield iterator
    # return (qbatch * 2, rbatch + negbatch)
    with open(qpath, 'rb') as f:
        qlen, qdataset = pickle.load(f)
        
    with open(rpath, 'rb') as f:
        rlen, rdataset = pickle.load(f)
        
    np.random.seed(seed)
    
    size = len(qdataset)    # dataset size
    idx = 0
    while True:
        qbatch = qdataset[idx:idx+batch_size]
        qll = qlen[idx:idx+batch_size]
        rbatch = rdataset[idx:idx+batch_size]
        rll = rlen[idx:idx+batch_size]
        
        pidx = np.random.choice(rdataset.shape[0], batch_size)
        nbatch = rdataset[pidx]
        nll = rlen[pidx]
        
        qbatch = np.concatenate([qbatch, qbatch])
        qll = np.concatenate([qll, qll])
        rbatch = np.concatenate([rbatch, nbatch])
        rll = np.concatenate([rll, nll])
        
        label = np.concatenate([np.ones(int(qbatch.shape[0] / 2)),
                                np.zeros(int(qbatch.shape[0] / 2))])
        
        # shuffle
        pureidx = np.arange(qbatch.shape[0])
        np.random.shuffle(pureidx)
        qbatch = qbatch[pureidx]
        qll = qll[pureidx]
        rbatch = rbatch[pureidx]
        rll = rll[pureidx]
        label = label[pureidx]
        
        idx += batch_size
        yield qbatch, rbatch, qll, rll, label
        
        if idx > size:
            break
    return None


def cal_avf_performance(path):
    su, sr, u = [], [], []
    # import ipdb
    # ipdb.set_trace()
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
    # Create the vocab file and embedding file for reference score
    # vocab and train id file
    parser = argparse.ArgumentParser(description='RUBER utils script')
    parser.add_argument('--dataset', type=str, default='xiaohuangji', 
                        help='the dataset we used')
    parser.add_argument('--mode', type=str, default='calculate', 
                        help='the mode for running the utils scripts (calculate|dataset)')
    args = parser.parse_args()
    
    if args.mode == 'dataset':
        process_train_file(f'./data/{args.dataset}/src-train.txt', 
                           f'./data/{args.dataset}/src-vocab.pkl',
                           f'./data/{args.dataset}/src-train-id.pkl')
        process_train_file(f'./data/{args.dataset}/tgt-train.txt', 
                           f'./data/{args.dataset}/tgt-vocab.pkl',
                           f'./data/{args.dataset}/tgt-train-id.pkl')
        process_train_file(f'./data/{args.dataset}/src-dev.txt', 
                           f'./data/{args.dataset}/src-vocab.pkl',
                           f'./data/{args.dataset}/src-dev-id.pkl',
                           mode='dev')
        process_train_file(f'./data/{args.dataset}/tgt-dev.txt', 
                           f'./data/{args.dataset}/tgt-vocab.pkl',
                           f'./data/{args.dataset}/tgt-dev-id.pkl',
                           mode='dev')
        process_train_file(f'./data/{args.dataset}/src-test.txt', 
                           f'./data/{args.dataset}/src-vocab.pkl',
                           f'./data/{args.dataset}/src-test-id.pkl',
                           mode='test')
        process_train_file(f'./data/{args.dataset}/tgt-test.txt', 
                           f'./data/{args.dataset}/tgt-vocab.pkl',
                           f'./data/{args.dataset}/tgt-test-id.pkl',
                           mode='test')
        
        word2vec, vec_dim, _ = load_word2vec('./embedding/glove.6B.300d.txt')
        make_embedding_matrix(f'./data/{args.dataset}/src-embed.pkl', 
                              word2vec, 
                              vec_dim, 
                              f'./data/{args.dataset}/src-vocab.pkl')
        make_embedding_matrix(f'./data/{args.dataset}/tgt-embed.pkl', 
                              word2vec, 
                              vec_dim, 
                              f'./data/{args.dataset}/tgt-vocab.pkl')
    elif args.mode == 'calculate':
        cal_avf_performance(f'./data/{args.dataset}/result.txt')
    else:
        print('[!] Wrong mode')
