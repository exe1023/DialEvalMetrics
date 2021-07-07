#pip3 install --user google-cloud-storage
#export STORAGE_BUCKET=
#export BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12

BERT_BASE_DIR=/usr0/home/yitingye/evaluations/bert_model/uncased_L-12_H-768_A-12

python3 create_tfrecord_data_from_json.py \
             --vocab_file=${BERT_BASE_DIR}/vocab.txt \
              --input_file=dataset/test.json \
              --output_file=dataset/test.tfrecord \
              --dupe_factor=1 --max_predictions_per_seq=0 --masked_lm_prob=0.001
