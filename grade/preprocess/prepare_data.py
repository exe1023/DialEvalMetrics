import os
import sys
import random
import json
import shutil
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
import copy
from transformers import *

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import torch
import torch.nn.functional as F
import texar.torch as tx
from texar.torch.modules import BERTEncoder, BERTEncoder


class PreprocessTool():
    def __init__(self):
        pass
    @staticmethod
    def print_save_file(filename):
        print("save result in {}".format(filename))
    
    @staticmethod
    def print_delete_file(filename):
        print("delete file {}".format(filename))

    @staticmethod
    def maybe_create_file(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _get_word2vec_numberbatch(self):
        output_info = "Convert glove embedding to word2vec embedding"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        glove_input_file = '../tools/numberbatch-en-19.08.txt'
        word2vec_output_file = '../tools/numberbatch-en-19.08.word2vec.txt'
        (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
        print(count, '\n', dimensions) 

    def _get_unique_original_ctx_rep(self):
        output_info = "Extract unique context-response pair in original dialogue"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        in_root = '../data/DailyDialog_tmp'
        output_root = '../data/DailyDialog'

        for mode in ['train', 'validation', 'test']:
            output_root_mode = '{}/{}'.format(output_root, mode)
            pair='pair-1'
            filename='original_dialog'
            output_root_mode_pair = '{}/{}'.format(output_root_mode, pair)
            PreprocessTool.maybe_create_file(output_root_mode_pair)

            text_data_path = '../data/{}/{}/{}/{}.text'.format(in_root, mode, pair, filename)
            text_set = set()
            o_text_data_path = '{}/{}.text'.format(output_root_mode_pair, filename)

            with open(text_data_path, 'r') as text_f, open(o_text_data_path, 'w') as o_text_f:
                for text_line in text_f.readlines():
                    if text_line.strip() in text_set:
                        continue
                    else:
                        text_line = text_line.strip()
                        text_set.add(text_line)
                        o_text_f.writelines([text_line + '\n'])
            PreprocessTool.print_save_file(o_text_data_path)
        
        print('Done!')
        print('\n')

    def _find_topK_hard_negative_response(self, rep, original_rep_embed_dict, candidate_embeddings_dict, mode):
        output_info = "Find topK hard negative response from candidates."
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        topk = 5
        random_num = 1000
        rep_embedding = torch.Tensor(original_rep_embed_dict[rep]).unsqueeze(0) 
        keys_chosen = random.sample(candidate_embeddings_dict.keys(), random_num)

        sims = []
        for idx, k in enumerate(keys_chosen):
            if k == rep:
                sims.append(-10000000.0)
                continue
            else:
                neg_embedding = torch.Tensor(candidate_embeddings_dict[k]).unsqueeze(0) 
                sims.append(F.cosine_similarity(rep_embedding, neg_embedding))

        sims = np.array(sims)
        top_k_neg_index = sims.argsort()[-topk:][::-1] 

        neg_list = []
        for idx, k in enumerate(keys_chosen):
            if idx in top_k_neg_index:
                neg_list.append(k)

        return neg_list

    @staticmethod
    def _save_rep_embed(original_rep_embed_dict, save_file):
        with open(save_file, 'w') as w_f:
            json.dump(original_rep_embed_dict, w_f)

    @staticmethod
    def _load_embedding_from_file(load_file):
        with open(load_file, 'r') as r_f:
            dict_result = json.load(r_f)

        return dict_result

    @staticmethod
    def _load_rep():
        root_dir = '../data/DailyDialog_tmp'
        train_file = '{}/train/pair-1/original_dialog_response.text'.format(root_dir)
        validation_file = '{}/validation/pair-1/original_dialog_response.text'.format(root_dir)
        test_file = '{}/test/pair-1/original_dialog_response.text'.format(root_dir)

        train_original_rep = []
        validation_original_rep = []
        test_original_rep = []
        with open(train_file, 'r') as train_f, open(validation_file, 'r') as validation_f, open(test_file, 'r') as test_f:
            for line in train_f.readlines():
                train_original_rep.append(line.strip())
            for line in validation_f.readlines():
                validation_original_rep.append(line.strip())
            for line in test_f.readlines():
                test_original_rep.append(line.strip())

        return train_original_rep, validation_original_rep, test_original_rep

    @staticmethod
    def _load_rep_embedding():
        device = 'cuda:1'
        max_rep_length = 28
        train_original_rep, validation_original_rep, test_original_rep = PreprocessTool._load_rep()
        train_original_rep_embed_dict = dict()
        validation_original_rep_embed_dict = dict()
        test_original_rep_embed_dict = dict()

        tokenizer = tx.data.BERTTokenizer(
            pretrained_model_name='bert-base-uncased')
        bert_encoder = BERTEncoder(
            pretrained_model_name='bert-base-uncased').to(device)

        for mode in ['train', 'validation', 'test']:
            if mode == 'train':
                original_rep = train_original_rep
            elif mode == 'validation':
                original_rep = validation_original_rep
            elif mode == 'test':
                original_rep = test_original_rep
            original_rep_embed_dict = dict()

            for _, item in tqdm(enumerate(original_rep), desc="loading {}'s rep embedding".format(mode)):
                item_cmpat = tx.utils.compat_as_text(item)
                input_ids, segment_ids, input_mask = tokenizer.encode_text(
                    text_a=item_cmpat,
                    max_seq_length=max_rep_length)
                input_ids = torch.Tensor(input_ids).to(device).unsqueeze(0).long()
                segment_ids = torch.Tensor(segment_ids).to(device).unsqueeze(0).long()
                input_mask = torch.Tensor(input_mask).to(device).unsqueeze(0).long()

                input_length = (1 - (input_ids == 0).int()).sum(dim=1)

                _, item_bert_embs = bert_encoder(
                    inputs = input_ids[:, 1:],
                    sequence_length = input_length-1,
                )

                original_rep_embed_dict[item] = np.array(item_bert_embs.squeeze(0).data.cpu()).tolist()

            if mode == 'train':
                train_original_rep_embed_dict = original_rep_embed_dict
                save_file = '../tools/train_original_rep_embed.txt'
            elif mode == 'validation':
                validation_original_rep_embed_dict = original_rep_embed_dict
                save_file = '../tools/validation_original_rep_embed.txt'
            elif mode == 'test':
                test_original_rep_embed_dict = original_rep_embed_dict
                save_file = '../tools/test_original_rep_embed.txt'

            PreprocessTool._save_rep_embed(original_rep_embed_dict, save_file)
            PreprocessTool.print_save_file(save_file)

        return train_original_rep_embed_dict, validation_original_rep_embed_dict, test_original_rep_embed_dict

    def _embedding_base_sampling(self):
        output_info = "Embedding base sampling"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        output_root = '../data/DailyDialog_tmp' 
        train_load_file = '../tools/train_original_rep_embed.txt'
        validation_load_file = '../tools/validation_original_rep_embed.txt'
        test_load_file = '../tools/test_original_rep_embed.txt'

        if not os.path.exists(train_load_file):
            train_original_rep_embed_dict, validation_original_rep_embed_dict, test_original_rep_embed_dict = \
                PreprocessTool._load_rep_embedding()
        else:
            train_original_rep_embed_dict = PreprocessTool._load_embedding_from_file(train_load_file)
            validation_original_rep_embed_dict = PreprocessTool._load_embedding_from_file(validation_load_file)
            test_original_rep_embed_dict = PreprocessTool._load_embedding_from_file(test_load_file)
        
        for mode in ['train', 'validation', 'test']:
            pair='pair-1'
            filename='original_dialog'

            if mode is 'train':
                original_rep_embed_dict = train_original_rep_embed_dict
            elif mode is 'validation':
                original_rep_embed_dict = validation_original_rep_embed_dict
            elif mode is 'test':
                original_rep_embed_dict = test_original_rep_embed_dict

            candidate_embeddings_dict = original_rep_embed_dict

            rep_data_path = '{}/{}/{}/{}_response.text'.format(output_root, mode, pair, filename)
            o_rep_data_path = '{}/{}/{}_embedding_neg.response'.format(output_root, mode, filename)
            with open(rep_data_path, 'r') as rep_f, open(o_rep_data_path, 'w') as w_f:
                for line in tqdm(rep_f.readlines(), desc='{}: Embedding base sampling...'.format(mode)):
                    rep = line.strip()
                    neg_list = self._find_topK_hard_negative_response(rep, original_rep_embed_dict, candidate_embeddings_dict, mode)
                    neg = '|||'.join(neg_list)
                    w_f.writelines([neg+'\n'])
            PreprocessTool.print_save_file(o_rep_data_path)
        print('Done!')
        print('\n')
    
    def _get_training_standard_data(self):
        output_info = "Get standard training dataset"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))
        
        i_root_dir = '../data/DailyDialog' 
        o_root_dir = '../data/DailyDialog'
        tmp_root_dir = '../data/DailyDialog_tmp'

        for mode in ['train', 'validation', 'test']:
            output_dir_1 = '{}/{}/pair-1'.format(o_root_dir, mode)
            output_dir_2 = '{}/{}/pair-2'.format(o_root_dir, mode)
            PreprocessTool.maybe_create_file(output_dir_2)

            i_filename = 'original_dialog'
            o_filename = 'perturbed_dialog'

            i_uni_text_data_path = '{}/{}/pair-1/{}.text'.format(i_root_dir, mode, i_filename)

            i_text_data_path = '{}/{}/pair-1/{}.text'.format(tmp_root_dir, mode, i_filename)
            i_key_data_path = '{}/{}/pair-1/{}.keyword'.format(tmp_root_dir, mode, i_filename)
            i_embedding_neg_rep_data_path = '{}/{}/{}_embedding_neg.response'.format(tmp_root_dir, mode, i_filename)
            i_lexical_neg_rep_data_path = '{}/{}/{}_lexical_neg.response'.format(tmp_root_dir, mode, i_filename)

            o_ori_text_data_path = '{}/{}.text'.format(output_dir_1, i_filename)
            o_ori_merge_keyword_data_path = '{}/{}_merge.keyword'.format(output_dir_1, i_filename)
            o_ori_keyword_data_path = '{}/{}.keyword'.format(output_dir_1, i_filename)
            o_ori_ctx_key_datapath = '{}/{}_merge.ctx_keyword'.format(output_dir_1, i_filename)
            o_ori_rep_key_datapath = '{}/{}_merge.rep_keyword'.format(output_dir_1, i_filename)
            o_per_text_data_path = '{}/{}.text'.format(output_dir_2, o_filename)
            o_per_merge_keyword_data_path = '{}/{}_merge.keyword'.format(output_dir_2, o_filename)
            o_per_keyword_data_path = '{}/{}.keyword'.format(output_dir_2, o_filename)
            o_per_ctx_key_datapath = '{}/{}_merge.ctx_keyword'.format(output_dir_2, o_filename)
            o_per_rep_key_datapath = '{}/{}_merge.rep_keyword'.format(output_dir_2, o_filename)

            # text_key_dict
            text_key_dict={}
            text_lexical_dict={}
            text_embedding_dict={}
            with open(i_text_data_path, 'r') as text_f, open(i_key_data_path, 'r') as key_f, \
                open(i_embedding_neg_rep_data_path, 'r') as embed_neg_rep_f, \
                open(i_lexical_neg_rep_data_path, 'r') as lexical_neg_rep_f:

                for _, (text_line, key_line, embed_negs, lexical_negs) in enumerate(zip(text_f.readlines(), \
                    key_f.readlines(), embed_neg_rep_f.readlines(), lexical_neg_rep_f.readlines())):
                    text_key_dict[text_line.strip().split('|||')[-1]] = key_line.strip().split('|||')[-1]
                    text_key_dict['|||'.join(text_line.strip().split('|||')[:-1])] = '|||'.join(key_line.strip().split('|||')[:-1])
                    text_lexical_dict[text_line.strip()] = lexical_negs.strip()
                    text_embedding_dict[text_line.strip()] = embed_negs.strip()

            ori_text_list = []
            ori_merge_keyword_list = []
            ori_keyword_list = []
            ori_ctx_key_list = []
            ori_rep_key_list = []
            per_text_list = []
            per_merge_keyword_list = []
            per_keyword_list = []
            per_ctx_key_list = []
            per_rep_key_list = []

            with open(i_uni_text_data_path, 'r') as uni_text_f:
                for uni_text_line in tqdm(uni_text_f.readlines(), desc="construct standard training data..."):
                    ctx = '|||'.join(uni_text_line.strip().split('|||')[:-1])
                    rep = uni_text_line.strip().split('|||')[-1]
                    neg_rep_list = text_lexical_dict[uni_text_line.strip()].split('|||')
                    for neg_item in neg_rep_list:
                        ori_text = ctx + '|||' + rep
                        per_text = ctx + '|||' + neg_item
                        ori_text_list.append(ori_text)
                        per_text_list.append(per_text)

                        ctx_key = text_key_dict[ctx]
                        rep_key = text_key_dict[rep]
                        if neg_item is not '':
                            neg_item_key = text_key_dict[neg_item]
                        else:
                            neg_item_key = ''

                        ori_merge_keyword = ' '.join((ctx_key + '|||' + rep_key).split('|||'))
                        ori_ctx_key = ' '.join(ctx_key.split('|||'))
                        ori_rep_key = rep_key
                        per_merge_keyword = ' '.join((ctx_key + '|||' + neg_item_key).split('|||'))
                        per_ctx_key = ' '.join(ctx_key.split('|||'))
                        per_rep_key = neg_item_key

                        ori_merge_keyword_list.append(ori_merge_keyword)
                        ori_keyword_list.append(ctx_key + '|||' + rep_key)
                        ori_ctx_key_list.append(ori_ctx_key)
                        ori_rep_key_list.append(ori_rep_key)
                        per_merge_keyword_list.append(per_merge_keyword)
                        per_keyword_list.append(ctx_key + '|||' + neg_item_key)
                        per_ctx_key_list.append(per_ctx_key)
                        per_rep_key_list.append(per_rep_key)
                        break

                    neg_rep_list = text_embedding_dict[uni_text_line.strip()].split('|||')
                    for neg_item in neg_rep_list:
                        ori_text = ctx + '|||' + rep
                        per_text = ctx + '|||' + neg_item
                        ori_text_list.append(ori_text)
                        per_text_list.append(per_text)

                        ctx_key = text_key_dict[ctx]
                        rep_key = text_key_dict[rep]
                        if neg_item is not '':
                            neg_item_key = text_key_dict[neg_item]
                        else:
                            neg_item_key = ''

                        ori_merge_keyword = ' '.join((ctx_key + '|||' + rep_key).split('|||'))
                        ori_ctx_key = ' '.join(ctx_key.split('|||'))
                        ori_rep_key = rep_key
                        per_merge_keyword = ' '.join((ctx_key + '|||' + neg_item_key).split('|||'))
                        per_ctx_key = ' '.join(ctx_key.split('|||'))
                        per_rep_key = neg_item_key

                        ori_merge_keyword_list.append(ori_merge_keyword)
                        ori_keyword_list.append(ctx_key + '|||' + rep_key)
                        ori_ctx_key_list.append(ori_ctx_key)
                        ori_rep_key_list.append(ori_rep_key)
                        per_merge_keyword_list.append(per_merge_keyword)
                        per_keyword_list.append(ctx_key + '|||' + neg_item_key)
                        per_ctx_key_list.append(per_ctx_key)
                        per_rep_key_list.append(per_rep_key)
                        break

            with open(o_ori_text_data_path, 'w') as ori_text_f, open(o_ori_merge_keyword_data_path, 'w') as ori_merge_key_f, \
                open(o_ori_keyword_data_path, 'w') as ori_key_f, \
                open(o_ori_ctx_key_datapath, 'w') as ori_ctx_key_f, open(o_ori_rep_key_datapath, 'w') as ori_rep_key_f, \
                open(o_per_text_data_path, 'w') as per_text_f, open(o_per_merge_keyword_data_path, 'w') as per_merge_key_f, \
                open(o_per_keyword_data_path, 'w') as per_key_f, \
                open(o_per_ctx_key_datapath, 'w') as per_ctx_key_f, open(o_per_rep_key_datapath, 'w') as per_rep_key_f:

                for _, (ori_text, ori_merge_keyword, ori_keyword, ori_ctx_key, ori_rep_key, \
                        per_text, per_merge_keyword, per_keyword, per_ctx_key, per_rep_key) in enumerate(zip( \
                        ori_text_list, ori_merge_keyword_list, ori_keyword_list, ori_ctx_key_list, ori_rep_key_list, \
                        per_text_list, per_merge_keyword_list, per_keyword_list, per_ctx_key_list, per_rep_key_list)):

                    ori_text_f.writelines([ori_text + '\n'])
                    ori_merge_key_f.writelines([ori_merge_keyword + '\n'])
                    ori_key_f.writelines([ori_keyword + '\n'])
                    ori_ctx_key_f.writelines([ori_ctx_key + '\n'])
                    ori_rep_key_f.writelines([ori_rep_key + '\n'])

                    per_text_f.writelines([per_text + '\n'])
                    per_merge_key_f.writelines([per_merge_keyword + '\n'])
                    per_key_f.writelines([per_keyword + '\n'])
                    per_ctx_key_f.writelines([per_ctx_key + '\n'])
                    per_rep_key_f.writelines([per_rep_key + '\n'])

            PreprocessTool.print_save_file(o_ori_text_data_path)
            PreprocessTool.print_save_file(o_ori_merge_keyword_data_path)
            PreprocessTool.print_save_file(o_ori_keyword_data_path)
            PreprocessTool.print_save_file(o_ori_ctx_key_datapath)
            PreprocessTool.print_save_file(o_ori_rep_key_datapath)
            PreprocessTool.print_save_file(o_per_text_data_path)
            PreprocessTool.print_save_file(o_per_merge_keyword_data_path)
            PreprocessTool.print_save_file(o_per_keyword_data_path)
            PreprocessTool.print_save_file(o_per_ctx_key_datapath)
            PreprocessTool.print_save_file(o_per_rep_key_datapath)
        
        print('Done!')
        print('\n')

    def _get_unique_keyword_pair(self):
        output_info = "Get unique keyword pair for find shortest path"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        data_name = 'DailyDialog'
        output='../data/{}/dialog_keyword_tuples.txt'.format(data_name)

        pre_word_list=[]
        tuples_nums_map = dict()

        for mode in ['train', 'validation', 'test']:
            for pair in ['pair-1', 'pair-2']:
                if pair=='pair-1':
                    filename='original_dialog'
                else:
                    filename='perturbed_dialog'
                with open("../data/{}/{}/{}/{}.keyword".format(data_name, mode, pair, filename), "r") as f:
                    next_keywords_list = [x.strip() for x in f.readlines()]
                    for keywords in tqdm(next_keywords_list):
                        keywords_list=[item.split(" ") for item in keywords.split("|||")]

                        # connect (c1_key,r1_key) (c2_key,r1_key) pair
                        for pre_idx in range(2):
                            pre_keywords_list = keywords_list[pre_idx]
                            post_keywords_list = keywords_list[2]
                            if pre_keywords_list[0]=='' or post_keywords_list[0]=='':
                                continue
                            for pre in pre_keywords_list:
                                for post in post_keywords_list:
                                    if (pre,post) not in tuples_nums_map:
                                        tuples_nums_map[(pre,post)]=1
                                    else:
                                        tuples_nums_map[(pre,post)]+=1
                f.close()
        with open(output, 'w') as w_f:
            for k,v in tuples_nums_map.items():
                pre_word_list.append(k[0])
                w_f.writelines([k[0]+"|||"+k[1]+'\n'])
        print(len(pre_word_list))
        PreprocessTool.print_save_file(output)

        print('Done!')
        print('\n')

    @staticmethod
    def load_cpnet():
        print("loading cpnet....")
        cpnet = nx.read_gpickle('../tools/cpnet.graph')
        print("Done")

        cpnet_simple = nx.MultiDiGraph()
        for u, v, data in cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                continue
            else:
                cpnet_simple.add_edge(u, v)
        
        return cpnet_simple
    
    @staticmethod
    def load_resources():
        concept2id = {}
        id2concept = {}
        with open("../tools/concept.txt", "r", encoding="utf8") as f:
            for w in f.readlines():
                concept2id[w.strip()] = len(concept2id)
                id2concept[len(id2concept)] = w.strip()

        print("concept2id done")
        id2relation = {}
        relation2id = {}
        with open("../tools/relation.txt", "r", encoding="utf8") as f:
            for w in f.readlines():
                id2relation[len(id2relation)] = w.strip()
                relation2id[w.strip()] = len(relation2id)
        print("relation2id done")

        return concept2id, id2concept, relation2id, id2relation

    def _find_shortest_hop_for_dialogue_keyword_tuple(self):
        output_info = "Find shortest path for keyword tuple"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))
        # hop: -1(No Path) 1(Self loop) 0(Do not find its edge) 2-n(Different levels of intimacy)

        i_datapath = "../data/DailyDialog/dialog_keyword_tuples.txt" 
        o_datapath = "../data/DailyDialog/dialog_keyword_tuples_multiGraph.hop"
        
        cpnet_simple = PreprocessTool.load_cpnet()
        concept2id, id2concept, relation2id, id2relation = PreprocessTool.load_resources()

        keyword_pair_set = []
        with open(i_datapath, 'r') as f, open(o_datapath, 'w') as w_f:
            for id, line in tqdm(enumerate(f.readlines())):
                source = line.strip().split("|||")[0]
                target = line.strip().split("|||")[1]

                source_tmp = source.replace("-", "_")
                target_tmp = target.replace("-", "_")
                if (source, target) not in keyword_pair_set:
                    keyword_pair_set.append((source, target))
                    if source_tmp not in concept2id or target_tmp not in concept2id:
                        w_f.writelines([source + "|||" + target + "|||" + str(-1) + '\n'])
                        continue

                    s = concept2id[source_tmp]
                    t = concept2id[target_tmp]
                    if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
                        w_f.writelines([source + "|||" + target + "|||" + str(-1) + '\n'])
                        continue
                    
                    try:
                        short_hop = nx.shortest_path_length(cpnet_simple, source=s, target=t) + 1 # compute shortest path
                    except:
                        short_hop = -1
                    w_f.writelines([source + "|||" + target + "|||" + str(short_hop) + '\n'])

        PreprocessTool.print_save_file(o_datapath)
        print('Done!')
        print('\n')
    
    def _get_training_data_keyword_vocab(self):
        output_info = "Get keyword vocab for training dataset"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        data_name = 'DailyDialog'
        vocab_set = set()
        o_datapath = '../data/{}/keyword.vocab'.format(data_name)
        w_f = open(o_datapath, 'w')
        for mode in ['train']:
            for pair in ['pair-1', 'pair-2']:
                if pair=='pair-1':
                    filename='original_dialog'
                else:
                    filename='perturbed_dialog'
                
                data_path = '../data/{}/{}/{}/{}.keyword'.format(data_name, mode, pair, filename)
                with open(data_path, 'r') as f:
                    for line in f.readlines():
                        keyword_list = ' '.join(line.strip().split('|||')).split(' ')
                        for item in keyword_list:
                            if item not in vocab_set:
                                vocab_set.add(item)
                                w_f.writelines([item + '\n'])
        w_f.close()
        PreprocessTool.print_save_file(o_datapath)
        print('Done!')
        print('\n')
    
    def _get_topk_one_hop_adjcent_for_every_keyword(self):
        output_info = "Find one hop neighbors for every keyword in keyword.vocab"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        o_data_name = 'DailyDialog_tmp'
        data_name = 'DailyDialog'
        word2vec_model_path = '../tools/numberbatch-en-19.08.word2vec.txt'
        word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path)
        concept2id, id2concept, relation2id, id2relation = PreprocessTool.load_resources()
        cpnet_simple = PreprocessTool.load_cpnet()
        print('load cpnet done!')

        for topk in [10,20]:
            o_datapath = '../data/{}/dialog_keyword.one_hop_{}_adjcent_node'.format(o_data_name, str(topk)) # refer to keyword.vocab
            i_datapath = '../data/{}/keyword.vocab'.format(data_name)

            with open(i_datapath, 'r') as f, open(o_datapath, 'w') as w_f:
                for line in tqdm(f.readlines()):
                    keyword = line.strip()
                    if keyword not in concept2id:
                        w_f.writelines(["" + '\n'])
                        continue

                    keyword_id = concept2id[keyword] # all edge include all nodes
                    if keyword_id not in cpnet_simple.nodes(): # not inside graph
                        w_f.writelines(["" + '\n'])
                        continue

                    neigh = list(cpnet_simple.neighbors(keyword_id))
                    neigh_keyword_list = []
                    neigh_sim = []

                    if len(neigh) == 0: # inside graph without any neighbors
                        w_f.writelines(["" + '\n'])
                        continue

                    if keyword not in word2vec_model.vocab:
                        for _, item in enumerate(neigh):
                            neigh_keyword = id2concept[item]
                            neigh_keyword_list.append(neigh_keyword)
                        if len(neigh_keyword_list) < topk:
                            topk_neigh = np.random.choice(neigh_keyword_list, topk, replace=True).tolist()
                        else:
                            topk_neigh = np.random.choice(neigh_keyword_list, topk, replace=False).tolist()
                    else:
                        for _, item in enumerate(neigh):
                            neigh_keyword = id2concept[item]
                            neigh_keyword_list.append(neigh_keyword)

                            if neigh_keyword not in word2vec_model.vocab:
                                similarity = -1000000
                                neigh_sim.append(similarity)
                            else:
                                similarity = word2vec_model.similarity(keyword, neigh_keyword)
                                neigh_sim.append(similarity)
                        
                        neigh_sim = np.array(neigh_sim)
                        neigh_keyword_list = np.array(neigh_keyword_list)
                        topk_index = neigh_sim.argsort()[-topk:][::-1] #from high to low

                        if len(topk_index.tolist()) < topk and len(topk_index.tolist()) > 0:
                            topk_index = np.random.choice(topk_index, topk, replace=True)
                        topk_neigh = neigh_keyword_list[topk_index].tolist()

                    w_f.writelines([' '.join(topk_neigh) + '\n'])
            
            PreprocessTool.print_save_file(o_datapath)
        print('Done!')
        print('\n')
    
    def _get_topk_only_two_hop_adjcent_for_every_keyword(self):
        output_info = "Find only two hop neighbors for every keyword in keyword.vocab"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        o_data_name = 'DailyDialog_tmp'
        data_name = 'DailyDialog'
        word2vec_model_path = '../tools/numberbatch-en-19.08.word2vec.txt'
        word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path)
        concept2id, id2concept, relation2id, id2relation = PreprocessTool.load_resources()
        cpnet_simple = PreprocessTool.load_cpnet()
        print('load cpnet done!')
        
        for topk in [10,20]:
            o_datapath = '../data/{}/dialog_keyword.only_two_hop_{}_adjcent_node'.format(o_data_name, str(topk))
            i_datapath = '../data/{}/keyword.vocab'.format(data_name)

            with open(i_datapath, 'r') as f, open(o_datapath, 'w') as w_f:
                for line in tqdm(f.readlines()):
                    keyword = line.strip()
                    if keyword not in concept2id:
                        w_f.writelines(["" + '\n'])
                        continue

                    keyword_id = concept2id[keyword] # all edge include all nodes
                    if keyword_id not in cpnet_simple.nodes():
                        w_f.writelines(["" + '\n'])
                        continue

                    one_hop_neigh = list(cpnet_simple.neighbors(keyword_id))
                    neigh = []
                    for item in one_hop_neigh:
                        if item in cpnet_simple:
                            item_neigh = list(cpnet_simple.neighbors(item))
                            for two_hop_item in item_neigh:
                                if two_hop_item == keyword_id:
                                    continue
                                neigh.append(two_hop_item)
                    neigh_keyword_list = []
                    neigh_sim = []

                    if len(neigh) == 0: # inside graph without neighbors
                        w_f.writelines(["" + '\n'])
                        continue
                    
                    if keyword not in word2vec_model.vocab:
                        for _, item in enumerate(neigh):
                            neigh_keyword = id2concept[item]
                            neigh_keyword_list.append(neigh_keyword)
                        if len(neigh_keyword_list) < topk:
                            topk_neigh = np.random.choice(neigh_keyword_list, topk, replace=True).tolist()
                        else:
                            topk_neigh = np.random.choice(neigh_keyword_list, topk, replace=False).tolist()
                    else:
                        for _, item in enumerate(neigh):
                            neigh_keyword = id2concept[item]
                            neigh_keyword_list.append(neigh_keyword)

                            if neigh_keyword not in word2vec_model.vocab:
                                similarity = -1000000
                                neigh_sim.append(similarity)
                            else:
                                similarity = word2vec_model.similarity(keyword, neigh_keyword)
                                neigh_sim.append(similarity)
                        
                        if len(neigh) == 0: # inside graph, no neighbors
                            w_f.writelines(["" + '\n'])
                            continue

                        neigh_sim = np.array(neigh_sim)
                        neigh_keyword_list = np.array(neigh_keyword_list)
                        topk_index = neigh_sim.argsort()[-topk:][::-1] # from high to low

                        if len(topk_index.tolist()) < topk and len(topk_index.tolist()) > 0:
                            topk_index = np.random.choice(topk_index, topk, replace=True)
                        topk_neigh = neigh_keyword_list[topk_index].tolist()

                    w_f.writelines([' '.join(topk_neigh) + '\n'])
            PreprocessTool.print_save_file(o_datapath)
        print('Done!')
        print('\n')

    def _add_only_neighbors_embedding(self):
        output_info = "Add all embedding of all neighbors"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))

        tmp_data_name = 'DailyDialog_tmp'
        data_name = 'DailyDialog'
        word2vec_model_path = '../tools/numberbatch-en-19.08.word2vec.txt'
        word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path)
        print("Finish Loading Embedding!")

        for _, (i_adj, o_keyword) in enumerate(zip( \
            ['dialog_keyword.only_two_hop_10_adjcent_node', 'dialog_keyword.only_two_hop_20_adjcent_node', \
            'dialog_keyword.one_hop_10_adjcent_node', 'dialog_keyword.one_hop_20_adjcent_node'], \
            ['2nd_hop_nr10.embedding', '2nd_hop_nr20.embedding', \
            '1st_hop_nr10.embedding', '1st_hop_nr20.embedding'])):

            i_adj_datapath = '../data/{}/{}'.format(tmp_data_name, i_adj)
            i_keyword_datapath = '../data/{}/keyword.vocab'.format(data_name)
            o_keyword_embedding = '../data/{}/{}'.format(data_name, o_keyword)
            
            line_num = 0
            with open(i_keyword_datapath, 'r') as keyword_f:
                for line in keyword_f.readlines():
                    line_num += 1
            
            str_list = []
            with open(i_adj_datapath, 'r') as adj_f, open(i_keyword_datapath, 'r') as keyword_f:

                id = 0
                for adj_line, keyword_line, in zip(adj_f.readlines(), keyword_f.readlines()):
                    hop_adjword_list = adj_line.strip().split(' ')
                    keyword = keyword_line.strip()

                    # add all neighbors embedding
                    embedding_added = 0
                    if keyword not in word2vec_model.vocab:
                        continue
                    keyword_embed = word2vec_model.get_vector(keyword)
                    new_keyword_embed = np.array([0.]*300)
                    for hop_word in hop_adjword_list:
                        if hop_word in word2vec_model.vocab:
                            hop_word_embed = copy.deepcopy(word2vec_model.get_vector(hop_word))
                            embedding_added += 1
                            new_keyword_embed += hop_word_embed
                    if embedding_added != 0:
                        new_keyword_embed = new_keyword_embed / embedding_added
                    else:
                        new_keyword_embed = np.zeros(300)

                    if id == 0:
                        print("np.round(new_keyword_embed,4): ", [round(i,4) for i in new_keyword_embed.tolist()])
                    new_keyword_embed_list = [round(i,4) for i in new_keyword_embed.tolist()]
                    new_keyword_embed_str = ' '.join(str(new_keyword_embed_list)[1:-1].split(', '))
                    cur_res = keyword + ' ' + new_keyword_embed_str
                    
                    str_list.append(cur_res)
                    id += 1

            df=pd.DataFrame(data=str_list, columns=['word_embedding'])
            df.to_csv(o_keyword_embedding, index=False,  header=False)
            PreprocessTool.print_save_file(o_keyword_embedding)
        
        tmp_root_dir = '../data/{}'.format(tmp_data_name)
        shutil.rmtree(tmp_root_dir) 
        PreprocessTool.print_delete_file(tmp_root_dir)
        print('Done!')
        print('\n')

    def _add_keyword_and_neighbors_embedding(self):
        output_info = "Add all embedding of all neighbors together with dialogue keyword"
        print('-' * len(output_info))
        print(output_info)
        print('-' * len(output_info))
        
        tmp_data_name = 'DailyDialog_tmp'
        data_name = 'DailyDialog'
        word2vec_model_path = '../tools/numberbatch-en-19.08.word2vec.txt'
        word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path)
        print("Finish Loading Embedding!")

        for _, (i_adj, o_keyword) in enumerate(zip( \
            ['dialog_keyword.one_hop_10_adjcent_node', 'dialog_keyword.one_hop_20_adjcent_node'], \
            ['one_hop_nr10_dialkw.embedding', 'one_hop_nr20_dialkw.embedding'])):

            i_adj_datapath = '../data/{}/{}'.format(tmp_data_name, i_adj) 
            i_keyword_datapath = '../data/{}/keyword.vocab'.format(data_name)
            o_keyword_embedding = '../data/{}/{}'.format(data_name, o_keyword) 
            
            line_num = 0
            with open(i_keyword_datapath, 'r') as keyword_f:
                for line in keyword_f.readlines():
                    line_num += 1
            
            str_list = []
            with open(i_adj_datapath, 'r') as adj_f, open(i_keyword_datapath, 'r') as keyword_f:
                id = 0
                for adj_line, keyword_line, in zip(adj_f.readlines(), keyword_f.readlines()):
                    hop_adjword_list = adj_line.strip().split(' ')
                    keyword = keyword_line.strip()

                    embedding_added = 0 # add all neighbors embedding
                    if keyword not in word2vec_model.vocab:
                        continue
                    keyword_embed = word2vec_model.get_vector(keyword)
                    new_keyword_embed = np.array([0.]*300)
                    for hop_word in hop_adjword_list:
                        if hop_word in word2vec_model.vocab:
                            hop_word_embed = copy.deepcopy(word2vec_model.get_vector(hop_word))
                            embedding_added += 1
                            new_keyword_embed += hop_word_embed
                    if embedding_added != 0:
                        new_keyword_embed = new_keyword_embed / embedding_added + copy.deepcopy(keyword_embed)
                    else:
                        new_keyword_embed = copy.deepcopy(keyword_embed)

                    if id == 0:
                        print("np.round(new_keyword_embed,4): ", [round(i,4) for i in new_keyword_embed.tolist()])
                    new_keyword_embed_list = [round(i,4) for i in new_keyword_embed.tolist()]
                    new_keyword_embed_str = ' '.join(str(new_keyword_embed_list)[1:-1].split(', '))
                    cur_res = keyword + ' ' + new_keyword_embed_str
                    
                    str_list.append(cur_res)
                    id += 1

            df=pd.DataFrame(data=str_list, columns=['word_embedding'])
            df.to_csv(o_keyword_embedding, index=False,  header=False)
            PreprocessTool.print_save_file(o_keyword_embedding)
    
if __name__ == "__main__":
    preprocess_tool = PreprocessTool()
    preprocess_tool._get_word2vec_numberbatch()                                 # numberbatch-en-19.08.word2vec.txt
    preprocess_tool._get_unique_original_ctx_rep()                              # attain unique context-response pair in pair-1
    preprocess_tool._embedding_base_sampling()                                  # embedding base sampling
    # IndexCreate.java  IndexSearch.java                                        # lucene lexical base sampling
    preprocess_tool._get_training_standard_data()                               # generate pair-2
    preprocess_tool._get_training_data_keyword_vocab()                          # get keyword vocab of training data

    preprocess_tool._get_unique_keyword_pair()                                  # get unique dialogue keyword tuple
    preprocess_tool._find_shortest_hop_for_dialogue_keyword_tuple()             # find shortest hop for each (ctx_key, rep_key) pair
    preprocess_tool._get_topk_one_hop_adjcent_for_every_keyword()               # get one hop neighbors (10 or 20)
    preprocess_tool._get_topk_only_two_hop_adjcent_for_every_keyword()          # get two hop neighbors (10 or 20)
    preprocess_tool._add_keyword_and_neighbors_embedding()                      # add only one-hop or only two-hop neighbors embedding + keyword embedding
    preprocess_tool._add_only_neighbors_embedding()                             # add only one-hop or only two-hop neighbors embedding 