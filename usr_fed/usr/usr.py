import random
import json
import csv
import numpy as np
import os
import regression
import spacy

import dr_api
import mlm_api
import arguments

# Eval prep
nlp = spacy.load('en')
def tokenize(data, max_length=500):
  new_data = []
  print("Tokenizing")
  data = [s.replace("_go", "").replace("_eos", "").strip() for s in data]
  docs = nlp.tokenizer.pipe([' '.join(s.lower().split()) for s in data])
  for doc in docs:
    # Tokenize with spacy
    tokenized = ' '.join([e.text for e in doc])

    # Fix mis-tokenized tags
    tokenized = "_go " + tokenized + " _eos"
    new_data.append(tokenized)

  return new_data

def prep_mlm(context, response):
  outputs = tokenize(response)
  valid_src = [e.strip().split("_eos ")[-1] for e in context]
  valid_src[0] = ' '.join(valid_src[0].split()[-52:])
 
  output_lines = [s + " _eos " + r + "\n" for s,r in zip(valid_src, outputs)]
  #open("undr/test.lm", "w+").writelines([' '.join(e.split()) + "\n" for e in output_lines])
  output_lines = [' '.join(e.split()) + "\n" for e in output_lines]
  
  with open('mlm_input.txt', 'a+') as f:
    f.writelines([' '.join(e.split()) + '\n' for e in output_lines])
  
  return ''.join(output_lines)

def prep_both(context, response, fact=None):
  #with open('both_shikib.tsv') as f:
  #  reader = csv.reader(f, delimiter='\t')
  #  rows = [row for row in reader]
  #return rows

  outputs = tokenize(response)
  valid_src = [e.strip().split("_eos ")[-1] for e in context]
  if fact:
    valid_fct = [e.strip() for e in fact]
  else:
    valid_fct = [e.strip() for e in context] 


  # restrit maximum length
  #input_len = len(valid_src[0].split() + valid_fct[0].split() + outputs[0].split())
  #exceed = input_len - 500
  #if exceed > 0:
  #  print(exceed)
  #  print(valid_src[0])
  #  valid_src[0] = ' '.join(valid_src[0].split()[exceed:])
  #  print(valid_src[0])
    
  valid_src[0] = ' '.join(valid_src[0].split()[-52:])

  valid_ctx = [s+" _eos " +f+" _eos" for s,f in zip(valid_src, valid_fct)]
  rows = [['0','1','2',c,o,'0'] for c,o in zip(valid_ctx, outputs)]

  with open('both_input.tsv', 'a+') as f:
    csv.writer(f, delimiter='\t').writerows(rows)
  rows = [rows[0]] + rows

  return rows

def prep_uk(context, response, fact=None):
  #with open('uk_shikib.tsv') as f:
  #  reader = csv.reader(f, delimiter='\t')
  #  rows = [row for row in reader]
  #return rows
  
  outputs = tokenize(response)

  if fact:
    valid_fct = [e.strip() for e in fact]
  else:
    valid_fct = [e.strip() for e in context] 

  valid_ctx = [f+" _eos" for f in valid_fct]

  rows = [['0','1','2',c,o,'0'] for c,o in zip(valid_ctx, outputs)]

  with open('uk_input.tsv', 'a+') as f:
    csv.writer(f, delimiter='\t').writerows(rows)

  rows = [rows[0]] + rows

  return rows

def init_args():
  # Here we handcraft where the pretrained models are
  #prefix = '/workspace/'
  prefix = '/usr0/home/yitingye/dialogue_metrics_docker/usr/'
  drc_args = arguments.arc_args(prefix + 'pretrained_models/ctx')
  drf_args = arguments.arf_args(prefix + 'pretrained_models/uk')
  mlm_args = arguments.mlm_args(prefix + 'pretrained_models/roberta_ft')
  '''
  prefix = '/usr0/home/ased/Code/usr/examples/'
  drc_args = arguments.arc_args(prefix + 'pc_both')
  drf_args = arguments.arf_args(prefix + 'pc_both')
  mlm_args = arguments.mlm_args(prefix + 'pc_both')
  '''
  return drc_args, drf_args, mlm_args

def init_models(drc_args, drf_args, mlm_args):
  drc_args, drc_model, drc_tokenizer = dr_api.init(drc_args)
  drf_args, drf_model, drf_tokenizer = dr_api.init(drf_args)
  mlm_args, mlm_model, mlm_tokenizer = mlm_api.init(mlm_args)

  return drc_args, drc_model, drc_tokenizer, \
         drf_args, drf_model, drf_tokenizer, \
         mlm_args, mlm_model, mlm_tokenizer


def get_dr_score(args, model, tokenizer, context, response, fact=None, model_type='drc'):
  if model_type == 'drc':
    data = prep_both(context, response, fact)
  elif model_type == 'drf':
    data = prep_uk(context, response, fact)
  else:
    raise Exception("No such dialog retrival metric. Choose from drc / drf")
  
  scores = dr_api.get_scores(args, model, tokenizer, data)
  return scores

def get_mlm_score(args, model, tokenizer, context, response):
  data = prep_mlm(context, response)
  scores = mlm_api.get_scores(args, model, tokenizer, data)
  return scores

def get_scores(context, response,
               drc_args, drc_model, drc_tokenizer,
               drf_args, drf_model, drf_tokenizer,
               mlm_args, mlm_model, mlm_tokenizer,
               fact=None):
  scores = {}
  drc_scores = get_dr_score(drc_args, drc_model, drc_tokenizer, 
                      context, response, fact, model_type='drc')
  drf_scores = get_dr_score(drf_args, drf_model, drf_tokenizer,
                      context, response, fact, model_type='drf')
  mlm_scores = get_mlm_score(mlm_args, mlm_model, mlm_tokenizer,
                      context, response)
  
  print(drc_scores, drf_scores, mlm_scores)
  scores['USR-DRc'], scores['USR-DRf'], scores['USR-MLM'] = np.mean(drc_scores), np.mean(drf_scores), np.mean(mlm_scores)
  # Regression
  regr_scores = regression.scores(mlm_scores, drc_scores, drf_scores)
  scores['USR'] = np.mean(regr_scores)
  print(scores)
  return scores
  
if __name__ == '__main__':
  drc_args, drf_args, mlm_args = init_args()
  drc_args, drc_model, drc_tokenizer, \
  drf_args, drf_model, drf_tokenizer, \
  mlm_args, mlm_model, mlm_tokenizer = init_models(drc_args, drf_args, mlm_args)
  scores = get_dr_score(drf_args, drf_model, drf_tokenizer,
               '', '', '', model_type='drc') 
  with open('both_shikib_scores.json', 'w') as f:
    json.dump(scores, f)
