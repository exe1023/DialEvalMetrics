MODEL_ID=na_all
MODEL_SAVE_DIR=full_acl_runs/
#DATA_LOC=eval_data/convai2_
DATA_LOC=$1
TRAIN_MODE=nce
VERSION=20488119

CUDA_VISIBLE_DEVICES=1 python codes/inference.py \
                       --id $MODEL_ID \
                       --model_save_dir $MODEL_SAVE_DIR \
                       --model_version $VERSION \
                       --train_mode nce \
                       --corrupt_pre $DATA_LOC \
                       --test_suffix '' \
                       --test_column true_response \
                       --results_file "maude_eval.jsonl"
