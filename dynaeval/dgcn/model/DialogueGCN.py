import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GCN import GCN
from .Classifier import Scorer
from .functions import batch_graphify
import dgcn

import random

log = dgcn.utils.get_logger()


class DynaEval(nn.Module):

    def __init__(self, args):
        super(DynaEval, self).__init__()

        self.g_dim = args.sentence_dim

#        ds1_dim = int(self.g_dim / 2)
        h0_dim = 300

        h1_dim = 150
        h2_dim = 150
        hc_dim = 150
        tag_size = 1

        self.args = args
        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.ds_1 = nn.Linear(self.g_dim, h0_dim)
#        self.ds_2 = nn.Linear(ds1_dim, h0_dim)
        self.drop = nn.Dropout(args.drop_rate)

        # self.bert = BertSeqContext(args)
        self.rnn = SeqContext(h0_dim, h0_dim, args)

        self.edge_att = EdgeAtt(h0_dim, args)
        self.gcn = GCN(h0_dim, h1_dim, h2_dim, args)

        self.clf = Scorer(h0_dim + h1_dim, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        # {'000':0, '001':1, '010':2, '011':3, '100':4, '101':5, '110':6, '111':7}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

        self.hinge = torch.nn.MarginRankingLoss(reduction='mean', margin=0.1).to(args.device)

    def get_rep(self, data):
        a_bs, a_num_utt, _ = data["a_text_tensor"].size()
        a_node_features = self.downsample(data["a_text_tensor"])
        a_node_features = self.rnn(data["a_text_len_tensor"], a_node_features)
        a_features, a_edge_index, a_edge_norm, a_edge_type, a_edge_index_lengths = batch_graphify(
            a_node_features, data["a_text_len_tensor"], data["a_speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)
        a_graph_out = self.gcn(a_features, a_edge_index, a_edge_norm, a_edge_type)

        b_bs, b_num_utt, _ = data["b_text_tensor"].size()
        b_node_features = self.downsample(data["b_text_tensor"])
        b_node_features = self.rnn(data["b_text_len_tensor"], b_node_features)
        b_features, b_edge_index, b_edge_norm, b_edge_type, b_edge_index_lengths = batch_graphify(
            b_node_features, data["b_text_len_tensor"], data["b_speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)
        b_graph_out = self.gcn(b_features, b_edge_index, b_edge_norm, b_edge_type)

        return a_graph_out, a_features, a_bs, a_num_utt, b_graph_out, b_features, b_bs, b_num_utt

    def downsample(self, data):
        ds_1 = self.drop(self.ds_1(data))
#        ds_2 = self.drop(self.ds_2(ds_1))
        return ds_1

    def forward(self, data):
        a_graph_out, a_features, a_bs, a_num_utt, b_graph_out, b_features, b_bs, b_num_utt = self.get_rep(data)

        a_lengths = data['a_text_len_tensor']
        a_new_features = torch.zeros((a_bs, a_num_utt, a_features.size(-1))).to(self.device)
        counter = 0
        for j in range(a_bs):
            a_cur_len = a_lengths[j].item()
            a_new_features[j, :a_cur_len, :] = a_features[counter:(counter + a_cur_len), :]
            counter += a_cur_len
        a_new_graph_out = torch.zeros((a_bs, a_num_utt, a_graph_out.size(-1))).to(self.device)
        counter = 0
        for j in range(a_bs):
            a_cur_len = a_lengths[j].item()
            a_new_graph_out[j, :a_cur_len, :] = a_graph_out[counter:(counter + a_cur_len), :]
            counter += a_cur_len

        b_lengths = data['b_text_len_tensor']
        b_new_features = torch.zeros((b_bs, b_num_utt, b_features.size(-1))).to(self.device)
        counter = 0
        for j in range(b_bs):
            b_cur_len = b_lengths[j].item()
            b_new_features[j, :b_cur_len, :] = b_features[counter:(counter + b_cur_len), :]
            counter += b_cur_len
        b_new_graph_out = torch.zeros((b_bs, b_num_utt, b_graph_out.size(-1))).to(self.device)
        counter = 0
        for j in range(b_bs):
            b_cur_len = b_lengths[j].item()
            b_new_graph_out[j, :b_cur_len, :] = b_graph_out[counter:(counter + b_cur_len), :]
            counter += b_cur_len

        a_coh = self.clf(torch.cat([a_new_features, a_new_graph_out], dim=-1), a_lengths)
        b_coh = self.clf(torch.cat([b_new_features, b_new_graph_out], dim=-1), b_lengths)

        rst = b_coh > a_coh

        rst = rst.long()

        return rst, a_coh

    def get_loss(self, data):
        a_graph_out, a_features, a_bs, a_num_utt, b_graph_out, b_features, b_bs, b_num_utt = self.get_rep(data)

        a_lengths = data['a_text_len_tensor']
        a_new_features = torch.zeros((a_bs, a_num_utt, a_features.size(-1))).to(self.device)
        counter = 0
        for j in range(a_bs):
            a_cur_len = a_lengths[j].item()
            a_new_features[j, :a_cur_len, :] = a_features[counter:(counter + a_cur_len), :]
            counter += a_cur_len
        a_new_graph_out = torch.zeros((a_bs, a_num_utt, a_graph_out.size(-1))).to(self.device)
        counter = 0
        for j in range(a_bs):
            a_cur_len = a_lengths[j].item()
            a_new_graph_out[j, :a_cur_len, :] = a_graph_out[counter:(counter + a_cur_len), :]
            counter += a_cur_len

        b_lengths = data['b_text_len_tensor']
        b_new_features = torch.zeros((b_bs, b_num_utt, b_features.size(-1))).to(self.device)
        counter = 0
        for j in range(b_bs):
            b_cur_len = b_lengths[j].item()
            b_new_features[j, :b_cur_len, :] = b_features[counter:(counter + b_cur_len), :]
            counter += b_cur_len
        b_new_graph_out = torch.zeros((b_bs, b_num_utt, b_graph_out.size(-1))).to(self.device)
        counter = 0
        for j in range(b_bs):
            b_cur_len = b_lengths[j].item()
            b_new_graph_out[j, :b_cur_len, :] = b_graph_out[counter:(counter + b_cur_len), :]
            counter += b_cur_len

        a_coh = self.clf(torch.cat([a_new_features, a_new_graph_out], dim=-1), a_lengths)
        b_coh = self.clf(torch.cat([b_new_features, b_new_graph_out], dim=-1), b_lengths)

        # coh_ixs is of the form [0,1,1,0,1], where 0 indicates the first one is the more coherent one
        # for this loss, the input is expected as [1,-1,-1,1,-1],
        # where 1 indicates the first to be coherent, while -1 the second
        # therefore, we need to transform the coh_ixs accordingly
        loss_coh_ixs = torch.add(torch.add(data["label_tensor"] * (-1),
                                           torch.ones(data["label_tensor"].size()).to(self.device)) * 2,
                                 torch.ones(data["label_tensor"].size()).to(self.device) * (-1))

        coh_loss = self.hinge(a_coh, b_coh, loss_coh_ixs)

        return coh_loss
