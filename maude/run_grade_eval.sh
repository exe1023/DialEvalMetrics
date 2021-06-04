DATA=$1
bash run_single.sh eval_data/${DATA}_grade_ bert_ranker
bash run_single.sh eval_data/${DATA}_grade_ dialogGPT
bash run_single.sh eval_data/${DATA}_grade_ transformer_generator
bash run_single.sh eval_data/${DATA}_grade_ transformer_ranker