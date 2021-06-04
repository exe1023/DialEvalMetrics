# Only Item to Modify
GPUs[0]=1
SEED=71
# SEED=$RANDOM

# Return to main directory
cd ../

# Train
bash ./script/train_sh/GRADE_K2_N10_N10.sh $SEED ${GPUs[0]} &