#DATA=(personachat_usr topicalchat_usr convai2_grade_bert_ranker convai2_grade_transformer_generator dailydialog_grade_transformer_ranker empatheticdialogues_grade_transformer_ranker convai2_grade_dialogGPT convai2_grade_transformer_ranker dailydialog_grade_transformer_generator  empatheticdialogues_grade_transformer_generator)
#for data in ${DATA[@]}
#do
#    echo "Eval $data"
#done
python hybird.py --dataset $1 --pretrain dailydialog
python hybird.py --dataset $1 --pretrain personachat