SEED=$1
OUTPUT_DIR=$SEED/GRADE_K2_N10_N10
METRIC_NAME=GRADE_K2_N10_N10_$SEED
MODEL_NAME=GRADE_K2_N10_N10
GPU=$2
DEVICES_ID=$2
TRAIN_BATCH_SIZE=16 
max_train_bert_epoch=5 

INIT_EMBD_FILE='./tools/numberbatch-en-19.08.txt' 
TRAINING_DATASET='DailyDialog' 
EVALUATOR='model_grade_K2'
CONFIG_DATA_FILE='./config/config_data_grade.py'
CONFIG_DATA_NAME='config_data_grade'

UNLIMIT_HOP=20 
TASK='train'
MODEL_FILE='model.evaluation_model.GRADE.'$EVALUATOR
PICKLE_DATA_DIR='./data/'$TRAINING_DATASET

python setting.py $INIT_EMBD_FILE $TRAIN_BATCH_SIZE $PICKLE_DATA_DIR $CONFIG_DATA_FILE  $max_train_bert_epoch 

python main_grade.py --do-train --config-data $CONFIG_DATA_NAME --model-file $MODEL_FILE --output-dir $OUTPUT_DIR --gpu $GPU --devices_id $DEVICES_ID --task $TASK \
--seed $SEED --metric-name $METRIC_NAME --dataset_dir $PICKLE_DATA_DIR --unlimit_hop $UNLIMIT_HOP --model_name $MODEL_NAME