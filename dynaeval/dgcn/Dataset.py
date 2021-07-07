import math
import random

import torch

# from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaModel

random.seed(2000)


class Dataset:
    def __init__(self, samples, args):
        self.samples = samples
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(self.samples) / args.batch_size)
        self.speaker_to_idx = {'A': 0, 'B': 1}
        self.max_seq_len = args.max_seq_len
        self.args = args
        self.sentence_model = RobertaModel.from_pretrained(args.model_name_or_path).to(args.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        self.sentence_dim = args.sentence_dim

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]
        self.in_batch_shuffle(batch)
        return batch

    def padding(self, samples):
        batch_size = len(samples)

        a_text_len_tensor = torch.tensor([len(s.text_1) for s in samples]).long()

        b_text_len_tensor = torch.tensor([len(s.text_2) for s in samples]).long()

        a_mx = torch.max(a_text_len_tensor).item()

        b_mx = torch.max(b_text_len_tensor).item()

        # text_tensor = {}
        #
        # if "roberta" not in self.model_type:
        #     token_type_ids = torch.zeros((batch_size, mx, self.max_seq_len)).long()
        #     input_ids = torch.zeros((batch_size, mx, self.max_seq_len)).long()
        # else:
        #     token_type_ids = None
        #     input_ids = torch.ones((batch_size, mx, self.max_seq_len)).long()
        #
        # attention_mask = torch.zeros((batch_size, mx, self.max_seq_len)).long()

        a_text_tensor = torch.zeros((batch_size, a_mx, self.sentence_dim))

        a_speaker_tensor = torch.zeros((batch_size, a_mx)).long()

        b_text_tensor = torch.zeros((batch_size, b_mx, self.sentence_dim))

        b_speaker_tensor = torch.zeros((batch_size, b_mx)).long()

        labels = []

        for i, s in enumerate(samples):
            # tokenized_list = [self.tokenizer(item,
            #                                  return_tensors='np',
            #                                  padding='max_length',
            #                                  truncation=True, max_length=self.max_seq_len) for item in s.text]
            #
            # cur_len = len(s.text)
            #
            # tmp = [torch.from_numpy(t['input_ids'][0]).long() for t in tokenized_list]
            # tmp = torch.stack(tmp)
            # input_ids[i, :cur_len, :] = tmp
            #
            # tmp = [torch.from_numpy(t['attention_mask'][0]).long() for t in tokenized_list]
            # tmp = torch.stack(tmp)
            # attention_mask[i, :cur_len, :] = tmp
            #
            # if "roberta" not in self.model_type:
            #     tmp = [torch.from_numpy(t['token_type_ids'][0]).long() for t in tokenized_list]
            #     tmp = torch.stack(tmp)
            #     token_type_ids[i, :cur_len, :] = tmp

            a_cur_len = len(s.text_1)

            b_cur_len = len(s.text_2)

            text_1 = [' '.join(item.split()[:self.args.max_seq_len]) + ' __eou__' for item in s.text_1]

            text_2 = [' '.join(item.split()[:self.args.max_seq_len]) + ' __eou__' for item in s.text_2]

            input_1 = {k:v.to(self.args.device) for k, v in self.tokenizer(text_1, padding=True, return_tensors="pt").items()}

            input_2 = {k:v.to(self.args.device) for k, v in self.tokenizer(text_2, padding=True, return_tensors="pt").items()}
            
            a_sentence_embeddings = self.sentence_model(**input_1)

            b_sentence_embeddings = self.sentence_model(**input_2)

            a_text_tensor[i, :a_cur_len, :] = torch.mean(a_sentence_embeddings[0].detach().cpu(), dim=1)

            a_speaker_tensor[i, :a_cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker_1])

            b_text_tensor[i, :b_cur_len, :] = torch.mean(b_sentence_embeddings[0].detach().cpu(), dim=1)

            b_speaker_tensor[i, :b_cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker_2])

            labels.append(s.label)

        label_tensor = torch.tensor(labels).long()

        # text_tensor['input_ids'] = input_ids  # [bs, max_a, max_seq]
        # text_tensor['attention_mask'] = attention_mask  # [bs, max_a, max_seq]
        # if "roberta" not in self.model_type:
        #     text_tensor['token_type_ids'] = token_type_ids  # [bs, max_a, max_seq]

        data = {
            "a_text_len_tensor": a_text_len_tensor,
            "a_text_tensor": a_text_tensor,
            "b_text_len_tensor": b_text_len_tensor,
            "b_text_tensor": b_text_tensor,
            "a_speaker_tensor": a_speaker_tensor,
            "b_speaker_tensor": b_speaker_tensor,
            "label_tensor": label_tensor
        }

        return data

    def in_batch_shuffle(self, samples):
        random.shuffle(samples)

    def shuffle(self):
        random.shuffle(self.samples)




