#!/bin/bash
# run.sh <mode> <dataset> <cuda> <model>
# for the convience of the repo: https://github.com/gmdftbyGMFTBY/MultiTurnDialogZoo.git
mode=$1
dataset=$2
model=$3
cuda=$4

if [ $mode = 'train' ]; then
    for i in {1..1}
    do
        echo "========== Iteration $i begins =========="
        CUDA_VISIBLE_DEVICES=$cuda python train_unreference.py \
            --dataset $dataset
    done
elif [ $mode = 'test' ]; then
    CUDA_VISIBLE_DEVICES=$cuda python hybird.py \
            --mode generate \
            --dataset $dataset \
            --model $model
elif [ $mode = 'ptest' ]; then
    # perturbation test for BERT-RUBER
    rm ./data/$dataset/$model-ptest.txt
    for i in {1..10}
    do
        CUDA_VISIBLE_DEVICES=$cuda python hybird.py \
            --mode generate \
            --dataset $dataset \
            --model $model-$i >> ./data/$dataset/$model-ptest.txt
    done
elif [ $mode = 'process' ]; then
    python utils.py \
            --mode $mode \
            --dataset $dataset
elif [ $mode = 'bertscore' ]; then
    CUDA_VISIBLE_DEVICES=$cuda python hybird.py \
            --mode bertscore \
            --dataset $dataset \
            --model $model >> result.txt
elif [ $mode = 'pbert' ]; then
    rm bert-ptest.txt
    CUDA_VISIBLE_DEVICES=$1 python hybird.py \
            --mode pbert >> bert-ptest.txt
fi
