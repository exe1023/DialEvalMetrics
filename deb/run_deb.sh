#export STORAGE_BUCKET=gs://ana_reddit_bucket
#BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12

BERT_BASE_DIR=/usr0/home/yitingye/evaluations/bert_model/uncased_L-12_H-768_A-12
DATADIR=$1

python3 create_tfrecord_data_from_json.py \
             --vocab_file=${BERT_BASE_DIR}/vocab.txt \
              --input_file=dataset/${1}/data.json \
              --output_file=dataset/${1}/data.tfrecord \
              --dupe_factor=1 --max_predictions_per_seq=0 --masked_lm_prob=0.001


python3 deb.py --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
             --init_checkpoint=deb_trained_on_rand_and_adv_neg/model.ckpt-102870 \
              --num_warmup_steps=100 \
              --input_file=dataset/${1}/data.tfrecord \
              --output_dir=dataset/${1} \
              --do_train=False \
              --max_eval_steps=475 \
               --use_tpu=False \
              --do_eval=False \
              --do_predict=True \
               --max_predictions_per_seq=0 

#python3 deb.py --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=${STORAGE_BUCKET}/deb_trained_on_rand_neg/model.ckpt-3214 --num_warmup_steps=100 --input_file=${STORAGE_BUCKET}/daily/DDpp_hard_neg_test.tfrecord --output_dir=${STORAGE_BUCKET}/daily/output --do_train=False --max_eval_steps=475 --use_tpu=True --tpu_name=tpu1 --do_eval=True --max_predictions_per_seq=0 
