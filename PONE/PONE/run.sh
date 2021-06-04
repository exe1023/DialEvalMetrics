#!/bin/bash
# $1: is the mode of running
# $2: is the dataset
# $3: is the gpuid
# $4: is the version for model
# ./run.sh epn tencent 0 1, run the epn-1 mode on tencent dataset and occupy the 0 GPU. 

# init the parameters
model_name=EPN-RUBER-${2}-${1}-${4}
dataset=$2
gpuid=$3
output_file="./data/$dataset/${model_name}-result.txt"
ckpt_dir="./ckpt/$dataset/$model_name"

echo "[!] Dataset: $dataset, Model: $model_name, Result file: $output_file, Ckpt: $ckpt_dir, GPU: $gpuid"

# clear ckpt and result file
rm $output_file
rm -rf $ckpt_dir
mkdir -p $ckpt_dir


if [ $1 = "origin" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== train without weight ==========
    # speed is fast, test more times
    for i in {1..1}
    do
        rm $ckpt_dir/*
        echo "========== Iteration $i begins =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --no-weight \
        --seed $(($i+10)) \
        --batch_size 256 \
        --lr 1e-3 \
        --patience 5 \
        --model_name $model_name \
        --src_train ./data/$dataset/src-train.embed \
        --tgt_train ./data/$dataset/tgt-train.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --dataset $dataset \
        --epoch_threshold 1
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --seed $(($i+10)) \
        --no-weight \
        --output ${output_file} \
        --model_name $model_name \
        --human_annotator ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/1/annotate.csv \
        --dataset $dataset \
        --epoch_threshold 1
    done
    
    # show the average performance
    echo "Average Performance on BERT-RUBER:"
    python utils.py --output $output_file --mode calculate
elif [ $1 = "da_origin" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== train with data augmentation ==========
    for i in {1..1}
    do
        rm $ckpt_dir/*
        echo "========== Iteration $i begins =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --no-weight \
        --da \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name $model_name \
        --src_train ./data/$dataset/src-train-da6.embed \
        --tgt_train ./data/$dataset/tgt-train-da6.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --dataset $dataset \
        --patience 5 \
        --epoch_threshold 1
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --seed $(($i+10)) \
        --no-weight \
        --output ${output_file} \
        --model_name $model_name \
        --human_annotator ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv ./data/annotator/$dataset/3/annotate.csv \
        --dataset $dataset \
        --epoch_threshold 1
    done
    
    # show the average performance
    echo "Average Performance on BERT-RUBER data augmentation:"
    python utils.py --output $output_file --mode calculate
elif [ $1 = "trda_origin" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== train with data augmentation ==========
    for i in {1..5}
    do
        rm $ckpt_dir/*
        echo "========== Iteration $i begins =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --no-weight \
        --da \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name $model_name \
        --src_train ./data/$dataset/src-train-da6.embed \
        --tgt_train ./data/$dataset/tgt-train-trda6.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --dataset $dataset \
        --patience 20 \
        --epoch_threshold 1
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --seed $(($i+10)) \
        --no-weight \
        --output ${output_file} \
        --model_name $model_name \
        --human_annotator ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv ./data/annotator/$dataset/3/annotate.csv \
        --dataset $dataset \
        --epoch_threshold 1
    done
    
    # show the average performance
    echo "Average Performance on BERT-RUBER data augmentation:"
    python utils.py --output $output_file --mode calculate
elif [ $1 = "da_weight" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== train with data augmentation ==========
    for i in {1..10}
    do
        rm $ckpt_dir/*
        echo "========== Iteration $i begins =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --weight \
        --da \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name $model_name \
        --src_train ./data/$dataset/src-train-da6.embed \
        --tgt_train ./data/$dataset/tgt-train-da6.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --dataset $dataset \
        --pre_choice 100 \
        --temp 0.1 \
        --weight_step 2 \
        --epoch_threshold 1 \
        --patience 20 \
        --weight_matrix ./data/$dataset/bert-weight-matrix.pkl \
        --epoch_threshold 1
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --seed $(($i+10)) \
        --no-weight \
        --output ${output_file} \
        --model_name $model_name \
        --human_annotator ./data/$dataset/person1-${dataset}-rest.txt ./data/$dataset/person2-${dataset}-rest.txt ./data/$dataset/person3-${dataset}-rest.txt \
        --dataset $dataset \
        --epoch_threshold 1
    done
    
    # show the average performance
    echo "Average Performance on BERT-RUBER data augmentation:"
    python utils.py --output $output_file --mode calculate
elif [ $1 = "trda_weight" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== train with data augmentation ==========
    for i in {1..10}
    do
        rm $ckpt_dir/*
        echo "========== Iteration $i begins =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --weight \
        --da \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name $model_name \
        --src_train ./data/$dataset/src-train-da6.embed \
        --tgt_train ./data/$dataset/tgt-train-trda6.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --dataset $dataset \
        --pre_choice 100 \
        --temp 0.05 \
        --weight_step 5 \
        --epoch_threshold 1 \
        --patience 20 \
        --weight_matrix ./data/$dataset/bert-weight-matrix.pkl \
        --epoch_threshold 1
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --seed $(($i+10)) \
        --no-weight \
        --output ${output_file} \
        --model_name $model_name \
        --human_annotator ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv ./data/annotator/$dataset/3/annotate.csv \
        --dataset $dataset \
        --epoch_threshold 1
    done
    
    # show the average performance
    echo "Average Performance on BERT-RUBER data augmentation:"
    python utils.py --output $output_file --mode calculate
elif [ $1 = "en" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== train with weight ==========
    for i in {1..1}
    do
        rm $ckpt_dir/*
        echo "========== Iteration $i begins =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --weight \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name ${model_name} \
        --bert_size 768 \
        --temp 0.05 \
        --pre_choice 100 \
        --patience 5 \
        --weight_matrix ./data/$dataset/bert-weight-matrix.pkl \
        --weight_step 3 \
        --epoch_threshold 1 \
        --src_train ./data/$dataset/src-train.embed \
        --tgt_train ./data/$dataset/tgt-train.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --no-bm25 \
        --dataset $dataset
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --seed $(($i+10)) \
        --weight \
        --output ${output_file} \
        --model_name ${model_name} \
        --epoch_threshold 1 \
        --human_annotator ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/1/annotate.csv \
        --dataset $dataset
    done
    # show the average performance
    echo "Average Performance on EPN-RUBER:"
    python utils.py --output $output_file --mode calculate
elif [ $1 = "ep" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== Ehancing Positive samples ==========
    rm ${output_file}_pretrained
    for i in {1..5}
    do
        echo "========== Iteration $i begins =========="
        rm -rf ${ckpt_dir}_pretrained
        mkdir -p ${ckpt_dir}_pretrained

        echo "========== Begin training pretrained model =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --no-weight \
        --da \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name ${model_name}_pretrained \
        --src_train ./data/$dataset/src-train.embed \
        --tgt_train ./data/$dataset/tgt-train.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --dataset $dataset \
        --patience 20 \
        --epoch_threshold 1
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --seed $(($i+10)) \
        --no-weight \
        --output ${output_file}_pretrained \
        --model_name ${model_name}_pretrained \
        --human_annotator ./data/annotator/$dataset/3/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv \
        --dataset $dataset \
        --epoch_threshold 1
        echo "========== End training pretrained model =========="

        rm ${ckpt_dir}/*
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --no-weight \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name ${model_name} \
        --pretrained_model ${model_name}_pretrained \
        --enhance \
        --lr 1e-7 \
        --weight_decay 0 \
        --bert_size 768 \
        --step 5 \
        --dataset $dataset \
        --epoch_threshold 1 \
        --epoches 100 \
        --src_train ./data/$dataset/src-train.embed \
        --tgt_train ./data/$dataset/tgt-train.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --da_src_train ./data/$dataset/src-train-da5.embed \
        --da_tgt_train ./data/$dataset/tgt-train-da5.embed \
        --temp 0.05 \
        --pre_choice 200 \
        --weight_step 2 \
        --fuzzy_threshold 0.5 \
        --patience 5 \
        --fuzzy_process drop \
        --weight_matrix ./data/$dataset/bert-weight-matrix.pkl
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --no-weight \
        --seed $(($i+10)) \
        --output $output_file \
        --model_name ${model_name} \
        --bert_size 768 \
        --dataset $dataset \
        --epoch_threshold 1 \
        --human_annotator ./data/annotator/$dataset/3/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv \

        echo "========== End training EPN-RUBER model weighted =========="
        echo "========== Iteration $i ends =========="
    done
    # show the average performance
    echo "Average Performance on pretrained models:"
    python utils.py --output ${output_file}_pretrained --mode calculate
    echo "Average Performance on EPN-RUBER-EP:"
    python utils.py --output $output_file --mode calculate
elif [ $1 = "trep" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== Ehancing Positive samples ==========
    rm ${output_file}_pretrained
    for i in {1..5}
    do
        echo "========== Iteration $i begins =========="
        rm -rf ${ckpt_dir}_pretrained
        mkdir -p ${ckpt_dir}_pretrained

        echo "========== Begin training pretrained model =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --da \
        --no-weight \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name ${model_name}_pretrained \
        --src_train ./data/$dataset/src-train-da6.embed \
        --tgt_train ./data/$dataset/tgt-train-trda6.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --dataset $dataset \
        --patience 20 \
        --epoch_threshold 1
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --seed $(($i+10)) \
        --no-weight \
        --output ${output_file}_pretrained \
        --model_name ${model_name}_pretrained \
        --human_annotator ./data/annotator/$dataset/3/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv \
        --dataset $dataset \
        --epoch_threshold 1
        echo "========== End training pretrained model =========="

        rm ${ckpt_dir}/*
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --no-weight \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name ${model_name} \
        --pretrained_model ${model_name}_pretrained \
        --enhance \
        --patience 5 \
        --lr 1e-7 \
        --weight_decay 1e-7 \
        --bert_size 768 \
        --step 2 \
        --dataset $dataset \
        --epoch_threshold 1 \
        --epoches 100 \
        --src_train ./data/$dataset/src-train.embed \
        --tgt_train ./data/$dataset/tgt-train.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --da_src_train ./data/$dataset/src-train-da5.embed \
        --da_tgt_train ./data/$dataset/tgt-train-trda5.embed \
        --temp 0.1 \
        --pre_choice 500 \
        --weight_step 2 \
        --fuzzy_threshold 0.05 \
        --fuzzy_process drop \
        --weight_matrix ./data/$dataset/bert-weight-matrix.pkl
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --no-weight \
        --seed $(($i+10)) \
        --output $output_file \
        --model_name ${model_name} \
        --bert_size 768 \
        --dataset $dataset \
        --epoch_threshold 1 \
        --human_annotator ./data/annotator/$dataset/3/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv \

        echo "========== End training EPN-RUBER model weighted =========="
        echo "========== Iteration $i ends =========="
    done
    # show the average performance
    echo "Average Performance on pretrained models:"
    python utils.py --output ${output_file}_pretrained --mode calculate
    echo "Average Performance on EPN-RUBER-EP:"
    python utils.py --output $output_file --mode calculate
elif [ $1 = "epn" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== Ehancing Positive samples ==========
    rm ${output_file}_pretrained
    for i in {1..1}
    do
        echo "========== Iteration $i begins =========="
        rm -rf ${ckpt_dir}_pretrained
        mkdir -p ${ckpt_dir}_pretrained

        echo "========== Begin training pretrained model =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --no-weight \
        --da \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name ${model_name}_pretrained \
        --src_train ./data/$dataset/src-train-da6.embed \
        --tgt_train ./data/$dataset/tgt-train-da6.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --dataset $dataset \
        --patience 5 \
        --epoch_threshold 1
        
        #CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        #--seed $(($i+10)) \
        #--no-weight \
        #--output ${output_file}_pretrained \
        #--model_name ${model_name}_pretrained \
        #--human_annotator ./data/annotator/$dataset/3/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv \
        #--dataset $dataset \
        #--epoch_threshold 1
        echo "========== End training pretrained model =========="

        rm ${ckpt_dir}/*
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --weight \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name ${model_name} \
        --pretrained_model ${model_name}_pretrained \
        --enhance \
        --lr 1e-6 \
        --weight_decay 0 \
        --bert_size 768 \
        --step 5 \
        --dataset $dataset \
        --epoch_threshold 1 \
        --epoches 75 \
        --src_train ./data/$dataset/src-train.embed \
        --tgt_train ./data/$dataset/tgt-train.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --da_src_train ./data/$dataset/src-train-da5.embed \
        --da_tgt_train ./data/$dataset/tgt-train-da5.embed \
        --temp 0.05 \
        --pre_choice 200 \
        --weight_step 2 \
        --fuzzy_threshold 0.05 \
        --fuzzy_process drop \
        --no-bm25 \
        --patience 5 \
        --weight_matrix ./data/$dataset/bert-weight-matrix.pkl
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --weight \
        --seed $(($i+10)) \
        --output $output_file \
        --model_name ${model_name} \
        --bert_size 768 \
        --dataset $dataset \
        --epoch_threshold 1 \
        --human_annotator ./data/annotator/$dataset/3/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv \

        echo "========== End training EPN-RUBER model weighted =========="
        echo "========== Iteration $i ends =========="
    done
    # show the average performance
    echo "Average Performance on pretrained models:"
    python utils.py --output ${output_file}_pretrained --mode calculate
    echo "Average Performance on EPN-RUBER:"
    python utils.py --output $output_file --mode calculate

elif [ $1 = "trepn" ]; then
    echo "========== Ready for training in mode ${1} =========="
    # ========== Ehancing Positive samples ==========
    rm ${output_file}_pretrained
    for i in {1..5}
    do
        echo "========== Iteration $i begins =========="
        rm -rf ${ckpt_dir}_pretrained
        mkdir -p ${ckpt_dir}_pretrained

        echo "========== Begin training pretrained model =========="
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --da \
        --no-weight \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name ${model_name}_pretrained \
        --src_train ./data/$dataset/src-train-da6.embed \
        --tgt_train ./data/$dataset/tgt-train-trda6.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --dataset $dataset \
        --patience 20 \
        --epoch_threshold 1
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --seed $(($i+10)) \
        --no-weight \
        --output ${output_file}_pretrained \
        --model_name ${model_name}_pretrained \
        --human_annotator ./data/annotator/$dataset/3/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv \
        --dataset $dataset \
        --epoch_threshold 1
        echo "========== End training pretrained model =========="
        # 0.08 (ft) / 3 (weight_step) / 0.1 (temp)

        rm ${ckpt_dir}/*
        CUDA_VISIBLE_DEVICES=$gpuid python train_unreference.py \
        --weight \
        --seed $(($i+10)) \
        --batch_size 256 \
        --model_name ${model_name} \
        --pretrained_model ${model_name}_pretrained \
        --enhance \
        --patience 5 \
        --lr 1e-7 \
        --weight_decay 1e-7 \
        --bert_size 768 \
        --step 2 \
        --dataset $dataset \
        --epoch_threshold 1 \
        --epoches 100 \
        --src_train ./data/$dataset/src-train.embed \
        --tgt_train ./data/$dataset/tgt-train.embed \
        --src_dev ./data/$dataset/src-dev.embed \
        --tgt_dev ./data/$dataset/tgt-dev.embed \
        --src_test ./data/$dataset/src-test.embed \
        --tgt_test ./data/$dataset/tgt-test.embed \
        --da_src_train ./data/$dataset/src-train-da5.embed \
        --da_tgt_train ./data/$dataset/tgt-train-trda5.embed \
        --temp 0.05 \
        --pre_choice 70 \
        --weight_step 2 \
        --fuzzy_threshold 0.05 \
        --fuzzy_process drop \
        --no-bm25 \
        --weight_matrix ./data/$dataset/bert-weight-matrix.pkl
        
        CUDA_VISIBLE_DEVICES=$gpuid python hybird.py \
        --weight \
        --seed $(($i+10)) \
        --output $output_file \
        --model_name ${model_name} \
        --bert_size 768 \
        --dataset $dataset \
        --epoch_threshold 1 \
        --human_annotator ./data/annotator/$dataset/3/annotate.csv ./data/annotator/$dataset/1/annotate.csv ./data/annotator/$dataset/2/annotate.csv \

        echo "========== End training EPN-RUBER model weighted =========="
        echo "========== Iteration $i ends =========="
    done
    # show the average performance
    echo "Average Performance on pretrained models:"
    python utils.py --output ${output_file}_pretrained --mode calculate
    echo "Average Performance on EPN-RUBER:"
    python utils.py --output $output_file --mode calculate
    
else
    echo "[!] Error, Unkown mode for training ..."
fi
