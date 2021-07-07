import torch
import torch.nn as nn
import torch.nn.functional as F

import dgcn

log = dgcn.utils.get_logger()


class Predictor(nn.Module):
    def __init__(self, input_dim, tag_size):
        super(Predictor, self).__init__()

        self.lin1 = nn.Linear(input_dim, tag_size)

    def forward(self, h):

        outputs = self.lin1(h)

        return outputs


class Scorer(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Scorer, self).__init__()

        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)

        if args.class_weight:
            self.loss_weights = torch.tensor([1 / 0.9, 1 / 0.1]).to(args.device)
            self.nll_loss = nn.NLLLoss(self.loss_weights)
        else:
            self.nll_loss = nn.NLLLoss()

    def get_score(self, h, text_len):
        hidden = self.drop(F.relu(self.lin1(h)))
        bs = text_len.size(0)
        a, b, c = hidden.size()
        device = h.get_device()
        reduced_hidden = torch.zeros((a, c)).to(device)
        for i in range(bs):
            t = torch.mean(hidden[i, :text_len[i].item(), :], dim=0)
            reduced_hidden[i, :] = t
        score = self.lin2(reduced_hidden).squeeze(1)
        return score

    def get_loss(self, label_tensor, h, text_len_tensor):
        log_prob = self.get_score(h, text_len_tensor)
        loss = self.nll_loss(log_prob, label_tensor)

        return loss

    def forward(self, h, text_len_tensor):

        coh = self.get_score(h, text_len_tensor)

        return coh





