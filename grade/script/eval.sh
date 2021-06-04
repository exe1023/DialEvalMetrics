# Only Item to Modify
GPUs[0]=1
SEED=71

# Return to main directory
cd ../

# Evaluate
bash ./script/eval_sh/Compute_GRADE_K2_N10_N10.sh $SEED ${GPUs[0]}
