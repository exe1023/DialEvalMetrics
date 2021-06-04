import sys
sys.path.append('./model/evaluation_model')
from typing import Optional

import torch
from torch import nn
import texar.torch as tx
import torch.nn.functional as F
from texar.torch.modules import WordEmbedder
from texar.torch.modules import MLPTransformConnector
from texar.torch.modules import BERTEncoder
import numpy as np
np.set_printoptions(threshold = 1e6)
from .model_util.GAT import GATLayer

class GRADE(nn.Module):
    """model code"""
    def __init__(self, args, model_config, data_config, embedding_init_value, device):
        super().__init__()

        self.config_model = model_config
        self.config_data = data_config
        self.vocab = tx.data.Vocab(self.config_data.vocab_file)

        self.bert_encoder = BERTEncoder(
            hparams=self.config_model.bert_encoder)

        self.linear0_1 = MLPTransformConnector(
            linear_layer_dim = 300*(16+2),
            output_size = 300)
        self.linear0_2 = MLPTransformConnector(
            linear_layer_dim = 300*(16+2),
            output_size = 300)
        self.linear0_3 = MLPTransformConnector(
            linear_layer_dim = 300*(16+2),
            output_size = 300)

        self.linear1 = MLPTransformConnector(
            linear_layer_dim = 300,
            output_size = 512)
        self.linear2_1 = MLPTransformConnector(
            linear_layer_dim = 768,
            output_size = 512)
        self.linear2_2 = MLPTransformConnector(
            linear_layer_dim = 768,
            output_size = 512)
        self.linear3 = MLPTransformConnector(
            linear_layer_dim = 1024,
            output_size = 512)
        self.linear4_1 = MLPTransformConnector(
            linear_layer_dim = 512,
            output_size = 128) 
        self.linear4_2 = MLPTransformConnector(
            linear_layer_dim = 512,
            output_size = 128) 
        self.linear5 = MLPTransformConnector(
            linear_layer_dim = 128,
            output_size = 1)
        
        self.word_embedder = WordEmbedder(
            vocab_size=self.vocab.size,
            init_value=embedding_init_value(1).word_vecs,  
            hparams=self.config_model.word_embedder_300
        )

        self.gat_1 = GATLayer(in_features=self.config_model.dim_c_300, 
            out_features=self.config_model.dim_c_300,
            alpha=0.2,
            nheads=4,
            activation=False,
            device=device)
        self.gat_2 = GATLayer(in_features=self.config_model.dim_c_300, 
            out_features=self.config_model.dim_c_300,
            alpha=0.2,
            nheads=4,
            activation=False,
            device=device)
        self.gat_3 = GATLayer(in_features=self.config_model.dim_c_300, 
            out_features=self.config_model.dim_c_300,
            alpha=0.2,
            nheads=4,
            activation=False,
            device=device)

        self.hinge = torch.nn.MarginRankingLoss(reduction='none', margin=0.1)

    def forward(self, mode,
        pair_1_input_ids_raw_text=None,
        pair_1_input_length_raw_text=None,
        pair_1_segment_ids_raw_text=None,
        pair_1_input_mask_raw_text=None,
        pair_1_batched_adjs=None,
        pair_1_input_ids_Keywords=None,
        pair_1_input_length_Keywords=None,
        pair_1_batch_onehop_embedding_matrix=None,
        pair_1_batch_twohop_embedding_matrix=None,

        pair_2_input_ids_raw_text=None, 
        pair_2_input_length_raw_text=None, 
        pair_2_segment_ids_raw_text=None, 
        pair_2_input_mask_raw_text=None, 
        pair_2_batched_adjs=None,
        pair_2_input_ids_Keywords=None,
        pair_2_input_length_Keywords=None,
        pair_2_batch_onehop_embedding_matrix=None,
        pair_2_batch_twohop_embedding_matrix=None,
        gt_preference_label=None,
        SCORES=False):
        
        score_of_pair_1 = self.get_score(pair_1_input_ids_raw_text, \
                                    pair_1_input_length_raw_text, \
                                    pair_1_segment_ids_raw_text, \
                                    pair_1_input_ids_Keywords, \
                                    pair_1_input_length_Keywords, \
                                    pair_1_batched_adjs)
        if SCORES:
            return score_of_pair_1.squeeze(1)

        score_of_pair_2 = self.get_score(pair_2_input_ids_raw_text, \
                                    pair_2_input_length_raw_text, \
                                    pair_2_segment_ids_raw_text, \
                                    pair_2_input_ids_Keywords, \
                                    pair_2_input_length_Keywords, \
                                    pair_2_batched_adjs)

        # calculate ranking loss
        pred_preference_label = (1 - (score_of_pair_1 > score_of_pair_2).int()).squeeze(1).int() 
        ranking_loss = self.RankingLoss(gt_preference_label, score_of_pair_1, score_of_pair_2)
        ranking_accu = tx.evals.accuracy(
            labels = gt_preference_label, 
            preds = pred_preference_label 
        )


        losses = ranking_loss

        return losses, ranking_loss, ranking_accu, pred_preference_label, score_of_pair_1, score_of_pair_2

    def RankingLoss(self, coh_ixs, coh1, coh2):
        """
        # coh_ixs is of the form [0,1,1,0,1], where 0 indicates the first one is the more coherent one
        # for this loss, the input is expected as [1,-1,-1,1,-1], where 1 indicates the first to be coherent, while -1 the second
        # therefore, we need to transform the coh_ixs accordingly
        """
        device = coh_ixs.get_device()
        loss_coh_ixs = torch.add(torch.add(coh_ixs*(-1), torch.ones(coh_ixs.size()).to(device))*2, torch.ones(coh_ixs.size()).to(device)*(-1))
        loss_coh = self.hinge(coh1, coh2, loss_coh_ixs)
        ranking_loss = loss_coh.mean()

        return ranking_loss

    def get_score(self, input_ids_raw_text, \
                        input_length_raw_text, \
                        segment_ids_raw_text, \
                        input_ids_keywords, \
                        input_length_Keywords, \
                        batched_adjs):

        device = input_ids_raw_text.get_device()
        batch_size = input_ids_raw_text.size(0)

        keyword_h_embed = self.word_embedder(input_ids_keywords) 

        # gat_1
        keyword_z_embed = self.gat_1(keyword_h_embed, batched_adjs) 
        keyword_h_embed = F.elu(self.linear0_1(keyword_h_embed.reshape(batch_size, -1)).reshape(batch_size, -1, 300) + keyword_z_embed) 
        # gat_2
        keyword_z_embed = self.gat_2(keyword_h_embed, batched_adjs) 
        keyword_h_embed = F.elu(self.linear0_2(keyword_h_embed.reshape(batch_size, -1)).reshape(batch_size, -1, 300) + keyword_z_embed) 
        # gat_3
        keyword_z_embed = self.gat_3(keyword_h_embed, batched_adjs) 
        keyword_h_embed = F.elu(self.linear0_3(keyword_h_embed.reshape(batch_size, -1)).reshape(batch_size, -1, 300) + keyword_z_embed) 
        

        keyword_h_embed = F.elu(self.linear1(torch.div(torch.sum(keyword_h_embed, dim=1), input_length_Keywords.unsqueeze(1)))) 

        _, bert_embed = self.bert_encoder(
            inputs = input_ids_raw_text[:,1:],
            sequence_length = input_length_raw_text-1,
            segment_ids = segment_ids_raw_text[:,1:]
        ) 
        bert_embed = self.linear2_1(bert_embed) 
        
        fusion_embs = torch.cat((bert_embed, keyword_h_embed), 1) 
        fusion_embs = F.elu(self.linear3(fusion_embs)) 

        linear = F.elu(self.linear4_1(fusion_embs)) 
        linear = self.linear5(linear)
        score = torch.sigmoid(linear)

        return score