### Steps to reproduce

A) Evaluate with MNLI

1. Put sampled data in 
    
       'self_logic/MNLI_SCORE/data/{}/to_label_{}.txt'.format(mode) 
   with the same format with mode in ['word_net','baseline','seq2seq'].
2. Run

       python self_logic/MNLI_SCORE/make_wordnet_para.py
   to make paired sentences for MNLI. Output 'examples/data/self_logic/{}/dev_matched.tsv'.format(mode), and 'self_logic/MNLI_SCORE/data/{}/samples.json'.format(mode)
3. run MNLI

       python run_glue.py --model_type roberta --model_name_or_path roberta-large-mnli --task_name MNLI --do_eval --do_lower_case --data_dir examples/data/self_logic/word_net --max_seq_length 128 --per_gpu_eval_batch_size=8 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir roberta_test1 --print_result --overwrite_cache
    with '--data_dir examples/data/self_logic/{}'.format(mode). mode in ['word_net','baseline','seq2seq']
    Output 'self_logic/MNLI_SCORE/data/{}/mnli.pred'.format(mode)
4. run
        
        python self_logic/MNLI_SCORE/compute_score.py
    to compute the auto-score from MNLI.
    Output 'self_logic/MNLI_SCORE/data/system_score_seq2seq.json'

B) Compare with human evaluation
Put human labeled files in 'self_logic/labeled'
run
        
        python self_logic/count_human_score.py
to print the scores.        

