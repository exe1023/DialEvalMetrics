#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
MODEL_DIR=~/tools/fairseq/wmt19.en-de.joined-dict.single_model
fairseq-interactive \
    --path $MODEL_DIR/model.pt $MODEL_DIR \
    --beam 5 --source-lang en --target-lang de \
    --tokenizer moses \
    --bpe fastbpe --bpe-codes $MODEL_DIR/bpecodes \
    --buffer-size 32 --input /checkpoint/koustuvs/dialog_metric/convai2_data/responses.txt >/checkpoint/koustuvs/dialog_metric/convai2_data/responses_de.txt
