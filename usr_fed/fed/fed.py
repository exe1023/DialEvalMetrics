import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

import math
from transformers import AutoTokenizer, AutoModelWithLMHead

# Old loading code. Use for from-scratch models
#tokenizer = GPT2Tokenizer.from_pretrained('dialogpt')
#model = GPT2LMHeadModel.from_pretrained('gpt2')
#weights = torch.load("dialogpt/small_fs.pkl")
#weights = {k.replace("module.", ""): v for k,v in weights.items()}
#weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
#weights.pop("lm_head.decoder.weight",None)
#model.load_state_dict(weights)


def load_models(name="microsoft/DialoGPT-large"):
  tokenizer = AutoTokenizer.from_pretrained(name)
  model = AutoModelWithLMHead.from_pretrained(name)
  return model, tokenizer

def score_batch(texts, tokenizer, model, batch_size=-1, max_seq_length=256, device='cpu'):
  '''
  texts: list of string
  tokenizer, model: pretrained tokenizer ana model from HuggingFace transformers
  batch_size: specify the batch size you want to use in inference. -1 means packing all queries in 1 batch.
  max_seq_length: specify the maximum sequence length after tokenization. Max: 1024
  device: which device to use. "cuda", "cpu"
  '''
  # make sure all text will in 1024:
  text_batchs = []
  for text in texts:
    tokenized = tokenizer.tokenize(text)
    if len(tokenized) > max_seq_length:
      tokenized = tokenized[-(max_seq_length):]
      tokenized[0] = tokenizer.eos_token # max sure we have special token at beginning.
    text_batchs.append(tokenized)

  # pad the input and generate attention mask
  pad_idx = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])
  token_ids = [tokenizer.convert_tokens_to_ids(s) for s in text_batchs]
  max_text_length = max([len(s) for s in token_ids])
  padded_tokens = [tok_ids + (pad_idx * (max_text_length - len(tok_ids))) for tok_ids in token_ids]
  input_ids = torch.tensor(padded_tokens)
  attention_mask = torch.zeros(input_ids.shape).long()
  for idx, tok_ids in enumerate(token_ids):
    attention_mask[idx][:len(tok_ids)] = 1

  model = model.to(device)
  input_ids = input_ids.to(device)
  attention_mask = attention_mask.to(device)

  with torch.no_grad():
      if batch_size == -1:
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs[1]
      else:
        logits = []
        for i in range(0, input_ids.size(0), batch_size):
          outputs = model(input_ids[i:i + batch_size, :], \
            attention_mask=attention_mask[i:i + batch_size, :], \
            labels=input_ids[i:i + batch_size, :])
          logits.append(outputs[1])
        logits = torch.cat(logits, dim=0)
  shifted_logits = logits[:, :-1, :].contiguous()
  labels = input_ids[:, 1:].contiguous()
  loss_fct = CrossEntropyLoss(reduction='none')
  lm_loss = loss_fct(shifted_logits.view(-1, model.config.vocab_size), labels.view(-1))

  return lm_loss.view(len(texts), -1)

def score(text, tokenizer, model):
  if not text.startswith("<|endoftext|> "):
    text = "<|endoftext|> " + text
  #input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
  tokenize_input = tokenizer.tokenize(text)

  if len(tokenize_input) >= 256:
    tokenize_input = ['<|endoftext|>'] + tokenize_input[-256:]
  #50256 is the token_id for <|endoftext|>
  tensor_input = torch.tensor([ tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
  with torch.no_grad():
      outputs = model(tensor_input, labels=tensor_input)
      loss, logits = outputs[:2]
  return loss.item()

def evaluate(conversation, model, tokenizer, truncate_type='normal'):
  scores = {}
  turn_level_utts = {
    "interesting": {
      "positive": ["Wow that is really interesting.", "That's really interesting!", "Cool! That sounds super interesting."],
      "negative": ["That's not very interesting.", "That's really boring.", "That was a really boring response."]
    },
    "engaging": {
      "positive": ["Wow! That's really cool!", "Tell me more!", "I'm really interested in learning more about this."],
      "negative": ["Let's change the topic.", "I don't really care. That's pretty boring.", "I want to talk about something else."]
    },
    "specific": {
      "positive": ["That's good to know. Cool!", "I see, that's interesting.", "That's a good point."],
      "negative": ["That's a very generic response.", "Not really relevant here.", "That's not really relevant here."]
    },
    "relevant": {
      "positive": [],
      "negative": ["That's not even related to what I said.", "Don't change the topic!", "Why are you changing the topic?"]
    },
    "correct": {
      "positive": [],
      "negative": ["You're not understanding me!", "I am so confused right now!", "I don't understand what you're saying."]
    },
    "semantically appropriate": {
      "positive": ["That makes sense!", "You have a good point."],
      "negative": ["That makes no sense!"]
    },
    "understandable": {
      "positive": ["That makes sense!", "You have a good point."],
      "negative": ["I don't understand at all!", "I'm so confused!", "That makes no sense!", "What does that even mean?"]
    },
    "fluent": {
      "positive": ["That makes sense!", "You have a good point."],
      "negative": ["Is that real English?", "I'm so confused right now!", "That makes no sense!"]
    },
  }

  if truncate_type == 'no_truncate':
    max_batch_size = 1
    max_seq_length = 1024
    device = 'cuda'
  elif truncate_type == 'normal':
    max_batch_size = 2
    max_seq_length = 128
    device = 'cuda'
  elif truncate_type == 'more':
    max_batch_size = 4
    max_seq_length = 64
    device = 'cuda'


  texts = []
  for metric, utts in turn_level_utts.items():
    pos, neg = utts["positive"], utts['negative']
    for m in pos:
      texts.append(conversation + " <|endoftext|> " + m)
    for m in neg:
      texts.append(conversation + " <|endoftext|> " + m)

  loss = score_batch(texts, tokenizer, model, batch_size=max_batch_size, max_seq_length=max_seq_length, device=device)
  idx = 0
  for metric, utts in turn_level_utts.items():
    pos, neg = utts["positive"], utts['negative']
    if len(pos) > 0:
      high_score = loss[idx: idx + len(pos), :].mean().item()
    else:
      high_score = 0
    idx += len(pos)
    if len(neg) > 0:
      low_score = loss[idx: idx + len(neg), :].mean().item()
    else:
      low_score = 0
    idx += len(neg)
    scores[metric] = (low_score - high_score)


  dialog_level_utts = {
    "coherent": {
      "positive": [],
      "negative": ["You're making no sense at all.", "You're changing the topic so much!", "You are so confusing."]
    },
    "error recovery": {
      "positive": [],
      "negative": ["I am so confused right now.", "You're really confusing.", "I don't understand what you're saying."]
    },
    "consistent": {
      "positive": [],
      "negative": ["That's not what you said earlier!", "Stop contradicting yourself!"],
    },
    "diverse": {
      "positive": [],
      "negative": ["Stop saying the same thing repeatedly.", "Why are you repeating yourself?", "Stop repeating yourself!"]
    },
    "depth": {
      "positive": [],
      "negative": ["Stop changing the topic so much.", "Don't change the topic!"],
    },
    "likeable": {
      "positive": ["I like you!", "You're super polite and fun to talk to", "Great talking to you."],
      "negative": ["You're not very nice.", "You're not very fun to talk to.", "I don't like you."]
    },
    "understand": {
      "positive": [],
      "negative": ["You're not understanding me!", "What are you trying to say?", "I don't understand what you're saying."]
    },
    "flexible": {
      "positive": ["You're very easy to talk to!", "Wow you can talk about a lot of things!"],
      "negative": ["I don't want to talk about that!", "Do you know how to talk about something else?"],
    },
    "informative": {
      "positive": ["Thanks for all the information!", "Wow that's a lot of information.", "You know a lot of facts!"],
      "negative": ["You're really boring.", "You don't really know much."],
    },
    "inquisitive": {
      "positive": ["You ask a lot of questions!", "That's a lot of questions!"],
      "negative": ["You don't ask many questions.", "You don't seem interested."],
    },
  }

  texts = []
  for metric, utts in dialog_level_utts.items():
    pos, neg = utts["positive"], utts['negative']
    for m in pos:
      texts.append(conversation + " <|endoftext|> " + m)
    for m in neg:
      texts.append(conversation + " <|endoftext|> " + m)
  loss = score_batch(texts, tokenizer, model, batch_size=max_batch_size, max_seq_length=max_seq_length, device=device)
  idx = 0
  for metric, utts in dialog_level_utts.items():
    pos, neg = utts["positive"], utts['negative']
    if len(pos) > 0:
      high_score = loss[idx: idx + len(pos), :].mean().item()
    else:
      high_score = 0
    idx += len(pos)
    if len(neg) > 0:
      low_score = loss[idx: idx + len(neg), :].mean().item()
    else:
      low_score = 0
    idx += len(neg)
    scores[metric] = (low_score - high_score)


  return scores
