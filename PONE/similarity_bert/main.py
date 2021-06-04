import os
from transformers import BertModel, BertConfig, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
from tqdm import tqdm
import ipdb
import numpy as np

'''
检查 bert embedding 的cosine距离
'''

def generate_attention_mask(inpt_ids):
    '''
    generate the corresponding attention mask according to the `input_ids`, which will 
    be fed into the model (BERT or GPT2)
    :inpt_ids: [batch, seq]
    
    return :attn_mask: [batch, seq]; 1 for not masked and 0 for masked tokens
    '''
    attn_mask = torch.zeros_like(inpt_ids)    # [batch, seq]
    not_masked_token_idx = inpt_ids.nonzero().transpose(0, 1).tolist()
    attn_mask[not_masked_token_idx] = 1
    # do not need the .cuda
    return attn_mask

def read_file(path):
    with open(path) as f:
        data = f.read().split('E\nM ')[1:]
        d = []
        for i in data:
            try:
                d.append(i.strip().split('\nM ')[1])
            except:
                d.append(i[2:])
    return d

def init_bert_model():
    model = BertModel.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    if torch.cuda.is_available():
        model.cuda()
    print(f'[!] init the bert model and tokenizer over ...')
    return model, tokenizer

def collate_fn(batch):
    batch = pad_sequence(batch, batch_first=True, padding_value=0)    # [batch, seq]
    if torch.cuda.is_available():
        batch = batch.cuda()
    return batch

class MineDataset(Dataset):
    
    def __init__(self, data, tokenizer):
        super(MineDataset, self).__init__()
        if os.path.exists('dataset_origin.pkl'):
            with open('dataset_origin.pkl', 'rb') as f:
                self.data = pickle.load(f)
            print(f'[!] load the dataset_origin from dataset_origin.pkl')
            return
        self.data = []
        for s in tqdm(data):
            s = tokenizer.encode(s)
            self.data.append(s)
        print(f'[!] collect {len(self.data)} samples for Bert')
        with open('dataset_origin.pkl', 'wb') as f:
            pickle.dump(self.data, f)
        print(f'[!] write the dataset_origin.pkl file')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return torch.LongTensor(self.data[i])

def packup(data, batch_size=32):
    dataloader = DataLoader(data, shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=collate_fn)
    return dataloader

def generate_bert_embeddings(model, dataloader):
    '''
    src is a list of string
    '''
    if os.path.exists('dataset.pkl'):
        print(f'[!] already have the dataset donot write the file')
        return None
    else:
        print(f'[!] begin to generate the bert embeddings')
    data = []
    for batch in tqdm(dataloader):
        attention_mask = generate_attention_mask(batch)
        batch = model(batch, attention_mask=attention_mask)[0]    # [batch, seq, 768]
        # ipdb.set_trace()
        batch = torch.mean(batch, dim=1).detach().cpu().numpy()    # [batch, 768]
        data.append(batch)
    data = np.concatenate(data)    # [data_size, 768]
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f'[!] save the dataset into dataset.pkl')
    
if __name__ == "__main__":
    data = read_file('xiaohuangji.txt')
    model, tokenizer = init_bert_model()
    dataset = MineDataset(data, tokenizer)
    dataloader = packup(dataset)
    
    generate_bert_embeddings(model, dataloader)
