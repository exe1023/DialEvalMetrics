BASEDIR=$1
METRIC=$2
DATA=$3
echo "Grade Metric"
echo "Correlation of bert_ranker"
python human_correlation.py $BASEDIR $METRIC bert_ranker $DATA
echo "Correlation of dialogGPT"
python human_correlation.py $BASEDIR $METRIC dialogGPT $DATA
echo "Correlation of transformer_generator"
python human_correlation.py $BASEDIR $METRIC transformer_generator $DATA
echo "Correlation of transformer_ranker"
python human_correlation.py $BASEDIR $METRIC transformer_ranker $DATA