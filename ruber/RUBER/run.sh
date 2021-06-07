#!/bin/bash

# ========== train with weight ==========
rm ./data/result.txt

for i in {5..5}
do
    echo "========== Iteration $i begins =========="
    CUDA_VISIBLE_DEVICES=$1 python train_unreference.py --lr 1e-5 --weight_decay 1e-6 --seed $((i+20)) --dataset $2
    CUDA_VISIBLE_DEVICES=$1 python hybird.py --dataset $2
done

python utils.py --dataset $2