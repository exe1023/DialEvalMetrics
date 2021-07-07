import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, RobertaModel, DistilBertModel


class SeqContext(nn.Module):

    def __init__(self, u_dim, g_dim, args):
        super(SeqContext, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim
        if args.rnn == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                               bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == "gru":
            self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                              bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, text_len_tensor, text_tensor):
        text_len_tensor = text_len_tensor.cpu().int()
        packed = pack_padded_sequence(
            text_tensor,
            text_len_tensor,
            batch_first=True,
            enforce_sorted=False
        )
        rnn_out, (_, _) = self.rnn(packed, None)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        return rnn_out


class BertSeqContext(nn.Module):

    def __init__(self, args):
        super(BertSeqContext, self).__init__()
        if args.bert_type == 'bert':
            self.bert = BertModel.from_pretrained(args.model_name_or_path)
        elif args.bert_type == "roberta":
            self.bert = RobertaModel.from_pretrained(args.model_name_or_path)
        else:
            self.bert = DistilBertModel.from_pretrained(args.model_name_or_path)

    def forward(self, text_tensor):

        bert_output = self.bert(**text_tensor)
        # [b * s, max_seq_len, dim]
        last_hidden_state = bert_output.last_hidden_state
        # max pooling
        pooled_output, pooled_index = torch.max(last_hidden_state, dim=1)

        return pooled_output
