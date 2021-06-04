#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.7.10

'''
utils file contains the tool function
1. bert embedding collect (query, groundtruth, generated)
2. load_best_model
3. batch iterator
'''

from es import *
from bm25_utils import *
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import argparse
import os
import re
import nltk
import jieba
import sys
import ipdb
from scipy.special import softmax
from bert_serving.client import BertClient
from itertools import combinations
from nltk.util import ngrams


def read_file(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
    

def cos_similarity(gr, ge):
    return np.dot(gr, ge) / (np.linalg.norm(gr) * np.linalg.norm(ge))


def normalize(scores):
    smin = min(scores)
    smax = max(scores)
    diff = smax - smin
    ret = [(s - smin) / diff for s in scores]
    return ret

def normalization_np(data):
    _range = np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True)
    return (data - np.min(data, axis=1, keepdims=True)) / _range

def get_best_model(dataset, model_name, net, threshold=30):
    file_path = f"./ckpt/{dataset}/{model_name}/best_model.ckpt"
    print(f'[!] Load the model from {file_path}, threshold {threshold}')
    net.load_state_dict(torch.load(file_path)['net']) 

def load_best_model(dataset, model_name, net, threshold=30):
    # set the limitation, not load the file that epoch below some threshold
    path = f"./ckpt/{dataset}/{model_name}/"
    best_acc, best_file = -1, None
    best_epoch = -1
    
    '''
    for file in os.listdir(path):
        try:
            _, acc, _, loss, _, epoch, _, weighted = file.split("_")
            epoch = epoch.split('.')[0]
        except:
            continue
        acc = float(acc)
        epoch = int(epoch)
        # if epoch > best_epoch:
        if epoch >= threshold and acc > best_acc:
            best_file = file
            # best_epoch = epoch
            best_acc = acc

    best_epoch = 0
    for file in os.listdir(path):
        _, acc, _, loss, _, epoch, _, weighted = file.split("_")
        epoch = epoch.split('.')[0]
        epoch = int(epoch)
        if epoch > best_epoch:
            best_file = file
            best_epoch = epoch
    '''
    best_file = 'best_model.ckpt'
    if best_file:
        file_path = path + best_file
        print(f'[!] Load the model from {file_path}, threshold {threshold}')
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception(f"[!] No saved model")


def get_batch(qpath, rpath, batch_size, 
              weighted=False, prechoice=200, t=0.1, 
              weight_matrix=None, weight_step=True, 
              seed=123, da=False, times=5, 
              fluent_path=None, safe_path=None, bm25=False, dataset='tencent'):
    # bert embedding matrix, [dataset_size, 3072]
    # return batch shape: [B, 3072]
    # weighted: True, weighted activated
    # prechoice: pre choose 5 times batch_size negative samples with random weights        
    qdataset = read_file(qpath)
    rdataset = read_file(rpath)
    np.random.seed(seed)
    if fluent_path:
        fluentdataset = read_file(fluent_path)
        print(f'[!] activate the fluent mode')
 
    if len(qdataset) != len(rdataset):
        used_len = min(len(qdataset), len(rdataset))
        qdataset = qdataset[:used_len]
        rdataset = rdataset[:used_len]
        
        print(f'Dataset length is different ! {used_len} {len(qdataset)} {len(rdataset)}')
       
        
    if safe_path:
        # 128, [45000, 128]
        safeidx, safeSRF = read_file(safe_path)
        # apply softmax convert to the probability
        safeSRF = normalization_np(safeSRF)
        safeSRF = torch.softmax(2 * torch.from_numpy(safeSRF), dim=1).numpy()
        safe_neg_whole = rdataset[safeidx]    # [128, 3072]
        print(f'[!] activate the safe mode: {safe_neg_whole.shape}')
        
        
    size = len(qdataset)
    idx = 0
    # esagent = ESChat('dailydialog') 
    if not weight_step and weighted and bm25:
        bm25model =  BM25Model(f'data/{dataset}/tgt-train.txt', threshold=5000)
        print(f'[!] process the {dataset}')

    with tqdm(total=len(qdataset)) as pbar:
        while True:
            qbatch = qdataset[idx:idx+batch_size]
            rbatch = rdataset[idx:idx+batch_size]
            
            if fluent_path:
                fidx = np.random.choice(fluentdataset.shape[0], len(rbatch))
                fluent_nbatch = fluentdataset[fidx]
                
            if safe_path:
                safe_nbatch = []
                if len(rdataset) == len(safeSRF):
                    for i in range(idx, idx+len(rbatch)):
                        safe_idx = np.random.choice(len(safeidx), p=safeSRF[i])
                        safe_nbatch.append(safe_neg_whole[safe_idx])
                    safe_nbatch = np.stack(safe_nbatch)
                else:
                    # positive data augmentation
                    # ipdb.set_trace()
                    dataset_size= int(len(rdataset) / (times+1))
                    part1 = times * dataset_size
                    for i in range(len(rbatch)):
                        if idx + i < part1:
                            safe_idx = np.random.choice(len(safeidx), p=safeSRF[(idx+i)//times])
                        else:
                            safe_idx = np.random.choice(len(safeidx), p=safeSRF[idx+i-part1])
                        safe_nbatch.append(safe_neg_whole[safe_idx])
                    safe_nbatch = np.stack(safe_nbatch)
            
            # weighted negative samples
            if not weight_step and weighted:
                # every sample need to re sample [prechoice] negative samples and calculate
                # speed may be slow, but this is necessary if the performance is better
                if da == False:
                    nbatch = []
                    for i in range(len(rbatch)):
                        if not bm25:
                            # BM25 Model donot need this line
                            prenidx = np.random.choice(weight_matrix.shape[0], prechoice)
                            weight_m = weight_matrix[idx+i, prenidx]    # [prechoice]
                            pren = rdataset[prenidx]
                            ssim = softmax(weight_m / t)                # [prechoice]
                            nidx = np.random.choice(ssim.shape[0], p=ssim)
                            nbatch.append(pren[nidx])
                        else:
                            # bm25 model
                            nidx = bm25model.get_weighted(idx+i)
                            nbatch.append(rdataset[nidx])
                    nbatch = np.stack(nbatch)    # [batch_size, 3072]
                else:
                    # two parts, first before times*dataset_size
                    dataset_size = int(len(qdataset) / (times+1))
                    part1 = times * dataset_size
                    real_tgt = rdataset[-dataset_size:]
                    nbatch = []
                    for i in range(len(rbatch)):
                        # 使用 bm25 作为补充实验
                        prenidx = np.random.choice(weight_matrix.shape[0], prechoice)
                        if idx+i < part1:
                            weight_m = weight_matrix[(idx+i)//times, prenidx]
                        else:
                            weight_m = weight_matrix[idx+i-part1, prenidx]
                        pren = real_tgt[prenidx]
                        ssim = softmax(weight_m / t)
                        nidx = np.random.choice(ssim.shape[0], p=ssim)
                        nbatch.append(pren[nidx])
                    nbatch = np.stack(nbatch)  
            else:
                pidx = np.random.choice(rdataset.shape[0], len(qbatch))
                nbatch = rdataset[pidx]
            
            # [!] add the context to fix the reviewers question
            # [!] add the random sample of the response to fix the reviewers question
            if fluent_path and not safe_path:
                qbatch = np.concatenate([qbatch, qbatch, qbatch])
                rbatch = np.concatenate([rbatch, nbatch, fluent_nbatch])
                
                label = np.concatenate([np.ones(int(qbatch.shape[0] / 3)),
                                        np.zeros(int(qbatch.shape[0] / 3)),
                                        np.zeros(int(qbatch.shape[0] / 3))])
            elif not fluent_path and safe_path:
                # ipdb.set_trace()
                qbatch = np.concatenate([qbatch, qbatch, qbatch])
                rbatch = np.concatenate([rbatch, nbatch, safe_nbatch])
                
                label = np.concatenate([np.ones(int(qbatch.shape[0] / 3)),
                                        np.zeros(int(qbatch.shape[0] / 3)),
                                        np.zeros(int(qbatch.shape[0] / 3))])
            elif fluent_path and safe_path:
                qbatch = np.concatenate([qbatch, qbatch, qbatch, qbatch])
                rbatch = np.concatenate([rbatch, nbatch, fluent_nbatch, safe_nbatch])
                
                label = np.concatenate([np.ones(int(qbatch.shape[0] / 4)),
                                        np.zeros(int(qbatch.shape[0] / 4)),
                                        np.zeros(int(qbatch.shape[0] / 4)),
                                        np.zeros(int(qbatch.shape[0] / 4))])
            else:
                qbatch = np.concatenate([qbatch, qbatch])
                rbatch = np.concatenate([rbatch, nbatch])
                
                label = np.concatenate([np.ones(int(qbatch.shape[0] / 2)),
                                        np.zeros(int(qbatch.shape[0] / 2))])
            
            
            # shuffle
            pureidx = np.arange(qbatch.shape[0])
            np.random.shuffle(pureidx)
            qbatch = qbatch[pureidx]
            rbatch = rbatch[pureidx]
            label = label[pureidx]
            
            idx += batch_size
            yield qbatch, rbatch, label
            pbar.update(batch_size)
            
            if idx > size:
                break
    return None


def get_batch_da(dalabel, spath, tpath, daspath, datpath, batch_size, 
                 weighted=False, prechoice=500, t=0.1, weight_matrix=None,
                 weight_step=True, seed=123, fuzzy_threshold=0.2, 
                 fuzzy_process='drop', dataset='tencent', bm25=False, fluent_path=None):
    '''get batch with data augmentation
    :param dalabel: the label of the augmentation data [0.11, 0.22, 0.9, 0.01, ...]
    :param daspath: data augmentation source path
    :param datpath: data augmentation target path
    :param dalabel: the label made by the RUBER
    :param word2vec: the word2vec is better than bert, path of the pkl file
    '''
    # source dataset: [S1, 3072], target dataset: [S1, 3072]
    # data augmentation souce dataset: [S2, 3072], target dataset: [S2, 3072]
    sd, td = read_file(spath), read_file(tpath)
    
    if fluent_path:
        fluentdataset = read_file(fluent_path)
    
    # process the augmentation data, fuzzy threshold to filter
    dasd, datd = read_file(daspath), read_file(datpath)
    filter_idx = np.where((dalabel <= fuzzy_threshold) | (dalabel >= (1 - fuzzy_threshold)))
    fuzzy_idx = np.where((dalabel > fuzzy_threshold) & (dalabel < (1 - fuzzy_threshold)))
    print(f'[!] fuzzy threshold: {fuzzy_threshold}, filter ratio: {round(len(filter_idx[0]) / len(dalabel), 4)}, fuzzy process mode: {fuzzy_process}')
    ab_dalabel = dalabel[filter_idx]
    fu_dalabel = dalabel[fuzzy_idx]
    ab_dalabel = (ab_dalabel > 0.5).astype(np.float)# make sure the 0 and 1 for training
    fu_dalabel = np.ones_like(fu_dalabel)   # all element is 1(positive samples)
    ab_dasd, ab_datd = dasd[filter_idx], datd[filter_idx]
    fu_dasd, fu_datd = dasd[fuzzy_idx], datd[fuzzy_idx]
    
    if fuzzy_process == 'change':
        dasd = np.concatenate([ab_dasd, fu_dasd])
        datd = np.concatenate([ab_datd, fu_datd])
        dalabel = np.concatenate([ab_dalabel, fu_dalabel])
    elif fuzzy_process == 'drop':
        dasd, datd, dalabel = ab_dasd, ab_datd, ab_dalabel
    else:
        raise Exception(f'[!] Fuzzy process mode {fuzzy_process} is wrong')
    # data augmentation process over
    
    daidx, idx = 0, 0
    dasize, size = len(dasd), len(sd)
    np.random.seed(seed)
    
    if not weight_step and weighted and bm25:
        bm25model =  BM25Model(f'data/{dataset}/tgt-train.txt', threshold=5000)
        print(f'[!] process the {dataset}')
    while True:
        real_qbatch = sd[idx:idx+batch_size]
        real_rbatch = td[idx:idx+batch_size]
        da_qbatch = dasd[daidx:daidx+batch_size]
        da_rbatch = datd[daidx:daidx+batch_size]
        da_label = dalabel[daidx:daidx+batch_size]
        
        if fluent_path:
            fidx = np.random.choice(fluentdataset.shape[0], len(real_qbatch))
            fluent_nbatch = fluentdataset[fidx]
        
        if not weight_step and weighted:
            nbatch = []
            for i in range(len(real_rbatch)):
                if not bm25:
                    prenidx = np.random.choice(weight_matrix.shape[0], prechoice)
                    weight_m = weight_matrix[idx+i, prenidx]    # [prechoice]
                    pren = td[prenidx]
                    ssim = softmax(weight_m / t)                # [prechoice]
                    nidx = np.random.choice(ssim.shape[0], p=ssim)
                    nbatch.append(pren[nidx])
                else:
                    # bm25model
                    nidx = bm25model.get_weighted(idx+i)
                    nbatch.append(rdataset[nidx])
            nbatch = np.stack(nbatch)                       # [batch_size, 3072]
        else:
            pidx = np.random.choice(td.shape[0], len(real_qbatch))
            nbatch = td[pidx]
        
        if fluent_path:
            qbatch = np.concatenate([real_qbatch, real_qbatch, real_qbatch, da_qbatch])
            rbatch = np.concatenate([real_rbatch, nbatch, fluent_nbatch, da_rbatch])

            label = np.concatenate([np.ones(len(real_qbatch)),
                                    np.zeros(len(real_qbatch)),
                                    np.zeros(len(real_qbatch)),
                                    da_label])
        else:
            qbatch = np.concatenate([real_qbatch, real_qbatch, da_qbatch])
            rbatch = np.concatenate([real_rbatch, nbatch, da_rbatch])

            label = np.concatenate([np.ones(len(real_qbatch)),
                                    np.zeros(len(real_qbatch)),
                                    da_label])
        
        # shuffle
        pureidx = np.arange(qbatch.shape[0])
        np.random.shuffle(pureidx)
        qbatch = qbatch[pureidx]
        rbatch = rbatch[pureidx]
        label = label[pureidx]
        
        idx += batch_size
        daidx += batch_size
        yield qbatch, rbatch, label
        
        if idx >= size:
            idx = 0
            
        if daidx >= dasize:
            break
            
    return None
    

def get_da_label(net, daspath, datpath, batch_size, fuzzy_threshold=0.2):
    # get the data augmentation label by the model
    # the label we return are the float, not the integer
    labels = []
    idx = 0
    q, r = read_file(daspath), read_file(datpath)
    size = len(q)
    while True:
        qbatch = q[idx:idx+batch_size]
        rbatch = r[idx:idx+batch_size]
        
        if torch.cuda.is_available():
            qbatch = torch.from_numpy(qbatch).cuda()
            rbatch = torch.from_numpy(rbatch).cuda()
        
        scores = net(qbatch, rbatch)    # [B]
        # label = (scores.cpu().detach().numpy() > 0.5).astype(np.float)
        label = scores.cpu().detach().numpy()
        labels.append(label)
        
        idx += batch_size
        
        if idx > size:
            break
    labels = np.concatenate(labels)    # [S]
    pratio = np.sum(labels) / len(labels)
    print(f'[!] get the label of augmentation data, positive ratio: {round(pratio, 2)}')
    return labels


# ========== fluent help function ========== #
def generate_fluent_negative_1(string):
    s = string.strip().split()
    random.shuffle(s)
    return s

def generate_fluent_negative_2(string):
    s = string.strip().split()
    try:
        for i in range(0, len(a), 2):
            s[i], s[i+1] = s[i+1], s[i]
    except:
        pass
    return s
    

def process_train_file(path, embed_path, batch_size=48, mode='NDA', da_size=6):
    # mode: NDA means No Data Augmentation
    # mode: DA means Data Augmentation, only for src file
    # batch_size: batch for bert to feedforward
    bc = BertClient()
    maxlength = 100
    dataset = []
    if mode == 'NDA':
        with open(path) as f:
            for line in f.readlines():
                if not line.strip():
                    line = '[UNK]'
                dataset.append(' '.join(line.strip().split()[-maxlength:]))
    elif mode == 'fluent':
        with open(path) as f:
            for idx, line in enumerate(f.readlines()):
                if not line.strip():
                    line = '<user0> <unk>'
                dataset.append(' '.join(generate_fluent_negative_1(line)))
                # dataset.append(' '.join(generate_fluent_negative_2(line)))
    elif mode == 'DA':
        with open(path) as f:
            for line in f.readlines():
                dataset.extend([' '.join(line.strip().split()[-maxlength:])])
        dataset = dataset * da_size
    elif mode == 'MT':
        # multi-turn
        with open(path) as f:
            for line in f.readlines():
                dataset.append(line.strip().split('__eou__')[-maxlength:])
    else:
        raise Exception('[!] Wrong mode')
    print('[!] load data over, begin to make bert embeddings')
        
    
    # bert-as-serive
    embed = []
    idx = 0
    if mode == 'MT':
        from itertools import accumulate
        while True:
            nbatch = dataset[idx:idx+batch_size]
            batch = []
            for i in nbatch:
                batch += i
            batch_length = list(accumulate([len(i) for i in nbatch]))
            batch_length = [0] + batch_length
            rest = bc.encode(batch)
            fr = []
            for i in range(1, len(batch_length)):
                fr.append(np.sum(rest[batch_length[i-1]:batch_length[i]], axis=0))
            embed.append(np.stack(fr))    # [b, 768]
            idx += batch_size
            if idx >= len(dataset):
                break
            print(f'{path}: {idx} / {len(dataset)}', end='\r')
        embed = np.concatenate(embed)
        print(f'embed shape: {embed.shape}')
    else:
        while True:
            batch = dataset[idx:idx+batch_size]
            rest = bc.encode(batch)    # [batch_size, 3072]
            embed.append(rest)
            idx += batch_size
            if idx >= len(dataset):
                break
            print(f'{path}: {idx} / {len(dataset)}', end='\r')
        embed = np.concatenate(embed)  # [dataset_size, 3072]

    with open(embed_path, 'wb') as f:
        pickle.dump(embed, f)
    print(f'Write the bert embedding into {embed_path}')
    print(f'Dataset size: {len(dataset)}')
        
        
def cal_avf_performance(path):
    def cal(su, u, mode):
        # cal avg performance
        avg_u_p, avg_u_s, avg_ruber_p, avg_ruber_s = [], [], [], []
        avg_u_pp, avg_u_ss, avg_ruber_pp, avg_ruber_ss = [], [], [], []
        for ku, ru in zip(su, u):
            avg_u_p.append(float(ku[0][0]))
            avg_u_pp.append(float(ku[0][1]))
            avg_u_s.append(float(ku[1][0]))
            avg_u_ss.append(float(ku[1][1]))
            avg_ruber_p.append(float(ru[0][0]))
            avg_ruber_pp.append(float(ru[0][1]))
            avg_ruber_s.append(float(ru[1][0]))
            avg_ruber_ss.append(float(ru[1][1]))
        print(f'{mode} Unrefer Avg pearson: {round(np.mean(avg_u_p), 5)}({round(np.mean(avg_u_pp), 5)}), Unrefer Avg spearman: {round(np.mean(avg_u_s), 5)}({round(np.mean(avg_u_ss), 5)})')
        print(f'{mode} RUBER Avg pearson: {round(np.mean(avg_ruber_p), 5)}({round(np.mean(avg_ruber_pp), 5)}), RUBER Avg spearman: {round(np.mean(avg_ruber_s), 5)}({round(np.mean(avg_ruber_ss), 5)})')
        
    su_f, sr_f, u_f = [], [], []
    su_c, sr_c, u_c = [], [], []
    su_s, sr_s, u_s = [], [], []
    su_o, sr_o, u_o = [], [], []
    with open(path) as f:
        p = re.compile('(0\.[0-9]+)\((.+?)\)')
        for line in f.readlines():
            m = p.findall(line.strip())
            if 'su_p_f' in line:
                su_f.append(m)
            elif 'su_p_c' in line:
                su_c.append(m)
            elif 'su_p_s' in line:
                su_s.append(m)
            elif 'su_p_o' in line:
                su_o.append(m)
            elif 'sr_p_f' in line:
                sr_f.append(m)
            elif 'sr_p_c' in line:
                sr_c.append(m)
            elif 'sr_p_s' in line:
                sr_s.append(m)
            elif 'sr_p_o' in line:
                sr_o.append(m)
            elif 'u_p_f' in line:
                u_f.append(m)
            elif 'u_p_c' in line:
                u_c.append(m)
            elif 'u_p_s' in line:
                u_s.append(m)
            elif 'u_p_o' in line:
                u_o.append(m)
            else:
                raise Exception("Wrong file format !")
                
    try:
        cal(su_f, u_f, 'fluency')
        cal(su_c, u_c, 'coherence')
        cal(su_s, u_s, 'safety')
    except:
        pass
    cal(su_o, u_o, 'overall')
    
    
def process_da_src_file(srcp, out, times=5):
    d = []
    with open(srcp) as f:
        for line in f.readlines():
            d.extend([line.strip()] * times)
    with open(out, 'w') as f:
        for line in d:
            f.write(line + '\n')
            
            
def get_weighted_matrix(trainrpath, savepath, max_size=50000):
    '''weighted matrix
    get the weighted matrix for the training dataset
    the max size of the matrix is limited to 50000 * 50000
    
    Use torch to accerlate
    
    bert is not good, try to use word2vec.
    '''
    # print('[!] Load the word2vec model')
    # print('[!] Load over')
    rd = read_file(trainrpath)
    if len(rd) > max_size:
        raise Exception(f"{len(rd)} is bigger than the limited max_size {max_size}")
    else:
        print(f'Size of matrix: [{len(rd)}, {len(rd)}]')
    
    # numpy version
    c = np.matmul(rd, rd.T)
    print('[!] over matmul')
    norm_c = np.linalg.norm(rd, axis=1)
    print('[!] over norm')
    cos_sim = c / norm_c.reshape(norm_c.shape[0], -1) / norm_c    # [len(rd), len(rd)]
    print('[!] over cos similarity')
    weight_m = cos_sim
    np.fill_diagonal(weight_m, 0)
    print('[!] over softmax')
        
    with open(savepath, 'wb') as f:
        pickle.dump(weight_m, f, protocol=4)
        
    print(f'Write the weight matrix into {savepath}')
    

def process_train_file_w2v(path, embedpath, modelpath):
    '''
    get the sentence vector of the utterence in the dataset
    bert is bad, so use this word2vec embedding to calculate the similarity
    '''
    import gensim
    print('[!] Load the model')
    model = gensim.models.KeyedVectors.load_word2vec_format(modelpath, binary=True)
    print('[!] Model load over')
    
    dataset = []
    with open(path) as f:
        for line in f.readlines():
            v = np.zeros(64)
            words = line.split()
            for word in words:
                if word in model:
                    v += model[word]
            v /= len(words)
            if np.sum(v) == 0:
                v = np.random.randn(64)
            dataset.append(v)
            
    embed = np.stack(dataset)    # [size, 64]
    
    with open(embedpath, 'wb') as f:
        pickle.dump(embed, f)
        
    print(f'Write the word2vec embedding into {embedpath}')
    print(f'Dataset size: {len(dataset)}')
    
    
def combine_da_data(dapath, opath, spath):
    # dapath: data augmentation data path, [5 * S, 3072]
    # opath: origin data path, [S, 3072]
    # spath: save path, [6 * S, 3072]
    da, o = read_file(dapath), read_file(opath)
    s = np.concatenate([da, o])
    
    with open(spath, 'wb') as f:
        pickle.dump(s, f)
        
    print(f'[!] Combine the data augmentation and origin data into {spath}')
    
    
# ========== process the dataset into the bert embeddings =========== # 
def process_dataset(times):
    # Generate the train / dev/ test / evluation data

    '''process_train_file(f'./data/{args.dataset}/src-train.txt', 
                       f'./data/{args.dataset}/src-train.embed')
    process_train_file(f'./data/{args.dataset}/tgt-train.txt', 
                       f'./data/{args.dataset}/tgt-train.embed')
    '''
    process_train_file(f'./data/{args.dataset}/tgt-train.txt', 
                        f'./data/{args.dataset}/tgt-train-fluent-neg.embed', 
                        mode='fluent')
    print(f'[!] Process the train dataset {args.dataset} over')
    # =========== dev and test do not need the data augmentation =========== #
    '''
    process_train_file(f'./data/{args.dataset}/src-dev.txt', 
                       f'./data/{args.dataset}/src-dev.embed')
    process_train_file(f'./data/{args.dataset}/tgt-dev.txt', 
                       f'./data/{args.dataset}/tgt-dev.embed')
    print(f'[!] Process the dev dataset {args.dataset} over')
    process_train_file(f'./data/{args.dataset}/src-test.txt', 
                       f'./data/{args.dataset}/src-test.embed')
    process_train_file(f'./data/{args.dataset}/tgt-test.txt', 
                       f'./data/{args.dataset}/tgt-test.embed')
    print(f'[!] Process the test dataset {args.dataset} over')
    

    process_train_file(f'./data/{args.dataset}/sample-100.txt', 
                       f'./data/{args.dataset}/sample-src.embed')
    process_train_file(f'./data/{args.dataset}/sample-100-tgt.txt', 
                       f'./data/{args.dataset}/sample-tgt.embed')
    process_train_file(f'./data/{args.dataset}/pred.txt', 
                       f'./data/{args.dataset}/pred.embed')
    print(f'[!] Process the evaluation dataset {args.dataset} over')
    
    
    # Generate the training data for Data Augumentation mode
    # tgt-train-da5.txt generated by OpenNMT-py
    process_da_src_file(f'./data/{args.dataset}/src-train.txt', 
                        f'./data/{args.dataset}/src-train-da{times}.txt', times=times)
    process_train_file(f'./data/{args.dataset}/src-train-da{times}.txt', 
                       f'./data/{args.dataset}/src-train-da{times}.embed')
    combine_da_data(f'./data/{args.dataset}/src-train-da{times}.embed', 
                    f'./data/{args.dataset}/src-train.embed', 
                    f'./data/{args.dataset}/src-train-da{times+1}.embed')
    
    try:
        process_train_file(f'./data/{args.dataset}/tgt-train-trda{times}.txt', 
                           f'./data/{args.dataset}/tgt-train-trda{times}.embed')
        print(f'[!] Process the data augmentation dataset {args.dataset} over')
        combine_da_data(f'./data/{args.dataset}/tgt-train-trda{times}.embed', 
                        f'./data/{args.dataset}/tgt-train.embed', 
                        f'./data/{args.dataset}/tgt-train-trda{times+1}.embed')
    except:
        print(f'[!] Can not find ./data/{args.dataset}/tgt-train-trda{times}.txt')
    #     
    # print(f'[!] Process the data augmentation dataset {args.dataset} over')
    # print(f'[!] only process the dataset {args.dataset}')
    process_train_file(f'./data/{args.dataset}/tgt-train-da{times}.txt', 
                       f'./data/{args.dataset}/tgt-train-da{times}.embed')
    combine_da_data(f'./data/{args.dataset}/tgt-train-da{times}.embed', 
                    f'./data/{args.dataset}/tgt-train.embed', 
                    f'./data/{args.dataset}/tgt-train-da{times+1}.embed')
    print(f'[!] Process the whole data augmentation over')
    '''
    # init the weight matrix for ehancing weight mode
    # fuck, 100k data occupied the motherf**ker 38 Gib space on disk, what the fuck !

    print(f'[!] process the motherfucker weighted negative sampler')
    get_weighted_matrix(f'./data/{args.dataset}/tgt-train.embed', 
                        f'./data/{args.dataset}/bert-weight-matrix.pkl', 
                        max_size=100000) 

    print(f'[!] Process weight matrix of dataset {args.dataset} over')
    print(f'[!] Process the {args.dataset} over')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EPN-RUBER utils script')
    parser.add_argument('--seed', type=int, default=123, 
                        help='seed for random init')
    parser.add_argument('--temp', type=float, default=0.1, 
                        help='the temp')
    parser.add_argument('--dataset', type=str, default='xiaohuangji', 
                        help='the dataset which need to be processed')
    parser.add_argument('--mode', type=str, default='calculate', 
                        help='the mode for running this script (calculate|process|da), process the dataset or calculate the average performance of the result.')
    parser.add_argument('--output', type=str, default='./data/xiaohuangji/result.txt', 
                        help='avg performance into the output file')
    parser.add_argument('--times', type=int, default=5, 
                        help='data augmentation times')
    args = parser.parse_args()
    if args.mode == 'calculate':
        print('========== Final result of this version ==========')
        cal_avf_performance(args.output)
        print('========== Final result of this version ==========')
    elif args.mode == 'process':
        process_dataset(args.times)
        #process_train_file(f'./data/{args.dataset}/tgt-train.txt', 
        #                f'./data/{args.dataset}/tgt-train-fluent-neg.embed', 
        #                mode='fluent')
    elif args.mode == 'process_infer':
        process_train_file(f'./data/{args.dataset}/context.txt', 
                        f'./data/{args.dataset}/context.embed')
        process_train_file(f'./data/{args.dataset}/reference.txt', 
                        f'./data/{args.dataset}/reference.embed')
        process_train_file(f'./data/{args.dataset}/hypothesis.txt', 
                        f'./data/{args.dataset}/hypothesis.embed') 
    elif args.mode == 'da':
        process_da_src_file(f'./data/{args.dataset}/src-train.txt', 
                            f'./data/{args.dataset}/src-train-da{args.times}.txt', 
                            times=args.times)
        print(f'[!] Process the **src** data augmentation dataset {args.dataset} over')
        process_train_file(f'./data/{args.dataset}/src-train-da{args.times}.txt', 
                           f'./data/{args.dataset}/src-train-da{args.times}.embed')
        process_train_file(f'./data/{args.dataset}/tgt-train-trda{args.times}.txt', 
                           f'./data/{args.dataset}/tgt-train-trda{args.times}.embed')
        print(f'[!] Process the data augmentation dataset {args.dataset} over')
        combine_da_data(f'./data/{args.dataset}/tgt-train-trda{args.times}.embed', 
                        f'./data/{args.dataset}/tgt-train.embed', 
                        f'./data/{args.dataset}/tgt-train-trda{args.times+1}.embed')
        combine_da_data(f'./data/{args.dataset}/src-train-da{args.times}.embed', 
                        f'./data/{args.dataset}/src-train.embed', 
                        f'./data/{args.dataset}/src-train-da{args.times+1}.embed')
        print(f'[!] Combine the data augmentation dataset {args.dataset} over')
    else:
        raise Exception('[!] Wrong mode for running ...')
