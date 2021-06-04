#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
MODEL_SAVE_DIR=full_acl_runs/
DATA_NAME=convai2
DATA_LOC=convai2_data/convai2_test_
FINE_TUNE_MODEL=convai2_data/distilbert_lm
TRAIN_MODE=nce
## Contains inference runs for the full tables in the paper.
## The MAUDE model is the only one uncommented for quick usage. (na_all)
# InferSent
# Train : All
# VERSION=20495233
# MODEL_ID=infersent_all
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --human_eval --results_file "human_eval.jsonl"
# InferSent
# Train: Only Semantics
# VERSION=20488124
# MODEL_ID=infersent_only_semantics
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# # Train: Only Syntax
# VERSION=20488125
# MODEL_ID=infersent_only_syntax
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# BertNLI
# Train : All
# VERSION=20497378
# MODEL_ID=bertnli_all
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --human_eval --results_file "human_eval.jsonl"
# BertNLI
# Train: Only Semantics
# VERSION=20488139
# MODEL_ID=bertnli_only_semantics
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat --results_file "bertnli.json"
# # # Train: Only Syntax
# VERSION=20488142
# MODEL_ID=bertnli_only_syntax
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order --results_file "bertnli.json"
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat --results_file "bertnli.json"
# MAUDE
# Train : All
VERSION=20488119
MODEL_ID=na_all
CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --human_eval --results_file "human_eval.jsonl"
# # Model
# # Train: Only Semantics
# VERSION=20488121
# MODEL_ID=na_only_semantics
# # CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# # CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# # CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# # CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# # # Train: Only Syntax
# VERSION=20488122
# MODEL_ID=na_only_syntax
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# VERSION=20488123
# MODEL_ID=na_all_context
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# RUBER
# Train : All
# VERSION=20688482
# MODEL_ID=ruber_all
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --human_eval --results_file "human_eval_ruber.jsonl"
# Train : Only Semantics
# MODEL_ID=ruber_only_semantics
# VERSION=0
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat --results_file "ruber.jsonl"
# # Train : Only Semantics
# MODEL_ID=ruber_only_syntax
# VERSION=0
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order --results_file "ruber.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat --results_file "ruber.jsonl"
## Zero Shot Transfer
## Dataset: Frames
## InferSent
# VERSION=20495233
# MODEL_ID=infersent_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/frames_data/fullframes_test_
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
## BertNLI
# VERSION=20497378
# MODEL_ID=bertnli_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/frames_data/fullframes_test_
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# ## Model
# VERSION=20488119
# MODEL_ID=na_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/frames_data/fullframes_test_
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# VERSION=20688482
# MODEL_ID=ruber_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/frames_data/fullframes_test_
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat --results_file "ruber_zero.jsonl"
## Dataset: MWOZ
# InferSent
# VERSION=20495233
# MODEL_ID=infersent_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/mwoz_data/fullmwoz_test_
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# # BertNLI
# VERSION=20497378
# MODEL_ID=bertnli_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/mwoz_data/fullmwoz_test_
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# ## Model
# VERSION=20488119
# MODEL_ID=na_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/mwoz_data/fullmwoz_test_
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# VERSION=20688482
# MODEL_ID=ruber_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/mwoz_data/fullmwoz_test_
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat --results_file "ruber_zero.jsonl"
# ## Dataset: DailyDialog
# # InferSent
# VERSION=20495233
# MODEL_ID=infersent_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/dailydialog_data/dailydialog_test_
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline infersent --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# # BertNLI
# VERSION=20497378
# MODEL_ID=bertnli_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/dailydialog_data/dailydialog_test_
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline bertnli --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# ## Model
# VERSION=20488119
# MODEL_ID=na_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/dailydialog_data/dailydialog_test_
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order
# CUDA_VISIBLE_DEVICES=0 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat
# VERSION=20688482
# MODEL_ID=ruber_all
# DATA_LOC=/checkpoint/koustuvs/dialog_metric/dailydialog_data/dailydialog_test_
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix true_response --test_column true_response --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix backtranslate --test_column backtranslate --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix seq2seq --test_column seq2seq --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix rand_utt_0 --test_column rand_utt --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix model_false_0 --test_column model_false --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_drop_0 --test_column word_drop --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_order_0 --test_column word_order --results_file "ruber_zero.jsonl"
# CUDA_VISIBLE_DEVICES=1 python codes/inference.py --id $MODEL_ID --model_save_dir $MODEL_SAVE_DIR  --model_version $VERSION --train_mode nce --train_baseline ruber --corrupt_pre $DATA_LOC --test_suffix word_repeat_0 --test_column word_repeat --results_file "ruber_zero.jsonl"
