models=(Seq2Seq HRED WSeq VHRED DSHRED MReCoSa HRAN DSHRED_RA)
rm result.txt
for model in ${models[@]}
do
    echo "========== eval $model =========="
    ./run.sh bertscore empchat $model $1
done
