DIALOG_DATASET_NAME=$1
DIALOG_MODEL_NAME=$2

# Ctx
DATAPREFIX=./evaluation/eval_data/$DIALOG_DATASET_NAME/$DIALOG_MODEL_NAME/
CTX_IN_FILENAME=human_ctx.txt
CTX_OUT_FILENAME=human_ctx.keyword

DATASET_DIR=./preprocess/dataset/$DIALOG_DATASET_NAME
IDF_PATH=$DATASET_DIR/idf.dict
CANDI_KW_PATH=$DATASET_DIR/candi_keywords.txt
INPUT_TEXT_PATH=$DATAPREFIX$CTX_IN_FILENAME
OUTPUT_KW_PATH=$DATAPREFIX$CTX_OUT_FILENAME

python ./preprocess/extract_keywords.py \
    --dataset_name $DIALOG_DATASET_NAME \
    --dataset_dir $DATASET_DIR \
    --idf_path $IDF_PATH \
    --candi_kw_path $CANDI_KW_PATH \
    --input_text_path $INPUT_TEXT_PATH \
    --kw_output_path $OUTPUT_KW_PATH

# Hyp
CTX_IN_FILENAME=human_hyp.txt
CTX_OUT_FILENAME=human_hyp.keyword
INPUT_TEXT_PATH=$DATAPREFIX$CTX_IN_FILENAME
OUTPUT_KW_PATH=$DATAPREFIX$CTX_OUT_FILENAME

python ./preprocess/extract_keywords.py \
    --dataset_name $DIALOG_DATASET_NAME \
    --dataset_dir $DATASET_DIR \
    --idf_path $IDF_PATH \
    --candi_kw_path $CANDI_KW_PATH \
    --input_text_path $INPUT_TEXT_PATH \
    --kw_output_path $OUTPUT_KW_PATH

