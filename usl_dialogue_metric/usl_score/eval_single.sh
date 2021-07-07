python predict.py \
    --weight-dir ../dailydialog/ \
    --context-file datasets/${1}/context.txt \
    --response-file datasets/${1}/hypothesis.txt \
    --output-score datasets/${1}/score.json 