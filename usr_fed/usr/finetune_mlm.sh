TRAIN_FILE=/home/yiting/dialogue_metrics_docker/test_data/lm_train.txt
DEV_FILE=/home/yiting/dialogue_metrics_docker/test_data/lm_dev.txt

python3 finetuning.py --finetune_task mlm \
    --per_gpu_train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --per_gpu_eval_batch_size=1 \
    --save_steps=1000 \
    --num_train_epochs=50 \
    --output_dir=mlm_finetune \
    --model_type=roberta \
    --model_name_or_path=robert-base \
    --train_data_file=$TRAIN_FILE \
    --do_train \
    --eval_data_file=$DEV_FILE \
    --mlm \
    --overwrite_output_dir \
