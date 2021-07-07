python eval_metric.py \
  --data_path ./datasets/${1}/data.json \
  --data_dir ./datasets/${1} \
  --metric fbd \
  --sample_num 10 \
  --model_type roberta-base \
  --batch_size 32
