#!/bin/bash

#SBATCH --job-name=train
#SBATCH -n 1
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -w hlt06
#SBATCH --output=log_dir/train.log

export dataset=empathetic
export dataset_dir=data/${dataset}
export task=us
export seed=345678
export lm_path=SRoBERTa

python -u train.py \
        --data=${dataset_dir}/${dataset}_${task}.pkl \
        --from_begin \
        --device=cuda \
        --epochs=20 \
        --batch_size=512 \
        --seed=${seed} \
        --wf=4 \
        --wp=4 \
        --model_name_or_path ${lm_path} \
        --model_save_path output/${dataset}-${task}-${seed}

