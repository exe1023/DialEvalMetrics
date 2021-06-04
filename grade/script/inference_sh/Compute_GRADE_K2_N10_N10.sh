SEED=$1
GPU=$2


DIALOG_DATASET_NAME=$3
DIALOG_MODEL_NAME=$4
HYP_FORMAT='hyp'
CTX_FORMAT='ctx'

MODEL_NAME=GRADE_K2_N10_N10
EVALUATOR_DIR=$SEED/GRADE_K2_N10_N10/
CKPT=model_eval_best_$SEED.ckpt
eval_metric_name=GRADE_K2_N10_N10_eval_best_$SEED
EVALUATOR='model_grade_K2'
unlimit_hop=20

gpu=$GPU
devices_id=$GPU
config_data='config_data_for_metric'
init_embd_file='./tools/numberbatch-en-19.08.txt'
data_dir='../data/'$DIALOG_DATASET_NAME
checkpoint='./output/'$EVALUATOR_DIR$CKPT
model_file='model.evaluation_model.GRADE.'$EVALUATOR
non_reduced_results_path='./evaluation/infer_result/'$DIALOG_DATASET_NAME'/'$DIALOG_MODEL_NAME'/non_reduced_results.json'
reduced_results_path='./evaluation/infer_result/'$DIALOG_DATASET_NAME'/'$DIALOG_MODEL_NAME'/reduced_results.json'

cd ./evaluation
python merge_keyword_and_text.py $DIALOG_MODEL_NAME $DIALOG_DATASET_NAME $HYP_FORMAT $CTX_FORMAT

cd ../preprocess/
python prepare_pkl_for_metric.py $data_dir $DIALOG_MODEL_NAME $DIALOG_DATASET_NAME $HYP_FORMAT $CTX_FORMAT $init_embd_file $config_data

cd ../
python main_for_metric_grade.py --do-metrics --checkpoint $checkpoint  --model-file $model_file --gpu $gpu --devices_id $devices_id --dialog-model-name $DIALOG_MODEL_NAME  \
--non_reduced_results_path $non_reduced_results_path --reduced_results_path $reduced_results_path --eval-metric-name $eval_metric_name --eval-dataset-name $DIALOG_DATASET_NAME \
--hyp-format $HYP_FORMAT --ctx-format $CTX_FORMAT --unlimit_hop $unlimit_hop --model_name $MODEL_NAME