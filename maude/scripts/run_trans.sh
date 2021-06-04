#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
MODEL_DIR=~/tools/fairseq/wmt19.en-de.joined-dict.single_model
DATASET=mwoz
MODE=test
CUDA_VISIBLE_DEVICES=0 python interactive_file.py \
    --path $MODEL_DIR/model.pt $MODEL_DIR \
    --beam 5 --source-lang en --target-lang de \
    --tokenizer moses \
    --bpe fastbpe --bpe-codes $MODEL_DIR/bpecodes \
    --buffer-size 64 --batch-size 64 \
    --input /checkpoint/koustuvs/dialog_metric/${DATASET}_data/${MODE}_response_inp.txt \
    --output_file /checkpoint/koustuvs/dialog_metric/${DATASET}_data/${MODE}_response_tmp.txt
MODEL_DIR=~/tools/fairseq/wmt19.de-en.joined-dict.single_model
CUDA_VISIBLE_DEVICES=0 python interactive_file.py \
    --path $MODEL_DIR/model.pt $MODEL_DIR \
    --beam 5 --source-lang de --target-lang en \
    --tokenizer moses \
    --bpe fastbpe --bpe-codes $MODEL_DIR/bpecodes \
    --buffer-size 64 --batch-size 64 \
    --input /checkpoint/koustuvs/dialog_metric/${DATASET}_data/${MODE}_response_tmp.txt \
    --output_file /checkpoint/koustuvs/dialog_metric/${DATASET}_data/${MODE}_response_back.txt
