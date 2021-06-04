source /usr0/home/yitingye/miniconda3/etc/profile.d/conda.sh

BASE=`pwd`
#DATA=(personachat_usr topicalchat_usr convai2_grade_bert_ranker convai2_grade_transformer_generator dailydialog_grade_transformer_ranker empatheticdialogues_grade_transformer_ranker convai2_grade_dialogGPT convai2_grade_transformer_ranker dailydialog_grade_transformer_generator  empatheticdialogues_grade_transformer_generator fed dstc6 dstc9)

#DATA=(convai2_grade_bert_ranker convai2_grade_transformer_generator dailydialog_grade_transformer_ranker empatheticdialogues_grade_transformer_ranker convai2_grade_dialogGPT convai2_grade_transformer_ranker dailydialog_grade_transformer_generator  empatheticdialogues_grade_transformer_generator)
#DATA=(dstc6 fed dstc9)
#DATA=(dstc6)
DATA=(fed engage fed_dialog dstc9)
#DATA=(engage)
#DATA=(holistic)
#DATA=(personachat_usr)
#DATA=(fed_dialog)
#DATA=(topicalchat_usr)
#DATA=(fed)

cd ${BASE}/FlowScore/
conda activate flow_eval
for data in ${DATA[@]}
do
    echo "Eval Flow"
    #python eval_single.py --eval_data eval_data/${data}.json --output results/${data}.json
done

# run amfm
cd ${BASE}/am_fm/examples/dstc6/
conda activate research
for data in ${DATA[@]}
do 
    echo "Eval AM-FM $data"
    #bash test_am.sh $data
    #bash test_lm.sh $data
done

# run maude
cd ${BASE}/maude/
conda activate research
for data in ${DATA[@]}
do
    echo "Eval Maude $data"
    #bash run_single.sh eval_data/${data}
done

# run holistic
cd ${BASE}/holistic_eval
conda activate hostilic_eval
for data in ${DATA[@]}
do
    echo "Eval holistic eval $data"
    #bash run_eval.sh $data
done


# run baseline
cd ${BASE}
conda activate research
for data in ${DATA[@]}
do
    echo "Eval Baseline $data"
    #python baseline_metric.py --data $data
done


# run ruber
cd ${BASE}/ruber_and_bert_ruber/RUBER
conda activate ruber_eval
for data in ${DATA[@]}
do
    echo "Eval RUBER $data"
    #bash run_eval.sh $data
done

# run bert_ruber + PONE
cd ${BASE}/PONE/PONE
conda activate research
for data in ${DATA[@]}
do
    echo "Eval BERT-RUBER + PONE $data"
    #bash run_eval.sh $data
done

# run grade
cd ${BASE}/grade
conda activate grade_eval
for data in ${DATA[@]}
do
    echo "Eval Grade $data"
    #bash run_single.sh $data
done

# run predictive_engagement
cd ${BASE}/predictive_engagement
conda activate research
for data in ${DATA[@]}
do
    echo "Eval predictive engagement $data"
    bash run_eval.sh $data
done

# run usr_fed
cd ${BASE}
conda activate research
for data in ${DATA[@]}
do
    echo "Eval USR FED $data"
    #python usr_fed_metric.py --data $data
done
