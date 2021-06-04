import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler


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
  model.to("cuda")
  return model, tokenizer

def score(text, tokenizer, model):
  if not text.startswith("<|endoftext|> "):
    text = "<|endoftext|> " + text
  input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
  tokenize_input = tokenizer.tokenize(text)
  #50256 is the token_id for <|endoftext|>
  tensor_input = torch.tensor([ tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
  with torch.no_grad():
      outputs = model(tensor_input, labels=tensor_input)
      loss, logits = outputs[:2]

  return loss.item() 

def evaluate(conversation, model, tokenizer):
  print(conversation)
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
  for metric,utts in turn_level_utts.items():
    pos = utts["positive"]
    neg = utts["negative"]

    # Positive score
    high_score = 0
    for m in pos:
      hs = score(conversation + " <|endoftext|> " + m, tokenizer, model) 
      high_score += hs 

    high_score = high_score/max(len(pos), 1)

    # Negative score
    low_score = 0
    for m in neg:
      ls = score(conversation + " <|endoftext|> " + m, tokenizer, model) 
      low_score += ls 
    low_score = low_score/max(len(neg), 1)

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
  for metric,utts in dialog_level_utts.items():
    pos = utts["positive"]
    neg = utts["negative"]

    # Positive
    high_score = 0
    for m in pos:
      hs = score(conversation + " <|endoftext|> " + m, tokenizer, model) 
      high_score += hs 

    high_score = high_score/max(len(pos), 1)

    # Negative
    low_score = 0
    for m in neg:
      ls = score(conversation + " <|endoftext|> " + m, tokenizer, model) 
      low_score += ls 
    low_score = low_score/max(len(neg), 1)

    scores[metric] = (low_score - high_score)

  return scores
