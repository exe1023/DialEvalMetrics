
python utils.py --dataset $1 --mode process_infer
python hybird.py --dataset $1 --model_name EPN-RUBER-dailydialog_sampled-epn-0 --pretrain_data dailydialog_sampled
python hybird.py --dataset $1 --model_name origin --pretrain_data dailydialog
python hybird.py --dataset $1 --model_name origin --pretrain_data personachat
