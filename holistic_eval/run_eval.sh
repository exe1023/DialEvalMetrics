

env CUDA_VISIBLE_DEVICES='' python metrics_evaluation.py --pretrained-model-path gpt2-pretrained-on-dailydialog/ --metric context --file-path eval_data/${1}.csv   --output-file-path eval_data/${1}_context_out.csv
env CUDA_VISIBLE_DEVICES='' python metrics_evaluation.py --pretrained-model-path gpt2-pretrained-on-dailydialog/ --metric fluency --file-path eval_data/${1}.csv   --output-file-path eval_data/${1}_fluency_out.csv
#env CUDA_VISIBLE_DEVICES='' python metrics_evaluation.py --pretrained-model-path gpt2-pretrained-on-dailydialog/ --metric diversity --file-path eval_data/${1}.csv   --output-file-path eval_data/${1}_diversity_out.csv
env CUDA_VISIBLE_DEVICES='' python metrics_evaluation.py --pretrained-model-path gpt2-pretrained-on-dailydialog/ --metric logic_consistency --file-path eval_data/${1}.csv   --output-file-path eval_data/${1}_logic_out.csv
