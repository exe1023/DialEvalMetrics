import os, sys
import codecs
import json
import shutil
from roberta_mnli.self_logic.MNLI_SCORE.make_wordnet_para_interface import make_wordnet_para_intf
from roberta_mnli.self_logic.MNLI_SCORE.compute_score_interface import compute_score_intf

def file_form(sentences_pre, sentences, inputf):
    wf = codecs.open(inputf, 'w', encoding='utf8')

    for dlg_index in range(len(sentences)):
        wf.write('sample '+str(dlg_index+1)+':\n')
        ss = sentences_pre[dlg_index].split('\n')
        wf.write('\thistory: ' + ss[0] + '\n')
        for s_index in range(1,len(ss)):
            wf.write('\t\t'+ss[s_index]+'\n')
        wf.write('\tpred: ' + sentences[dlg_index] + '\n')
        wf.write('\tYOUR SCORE:\n')
    wf.close()

def logic_eval(sentences_pre, sentences):
    mode = 'temp'
    # Put sampled data in 'self_logic/MNLI_SCORE/data/{}/to_label_{}.txt'.format(mode)
    intermef = os.path.join('roberta_mnli', 'self_logic', 'MNLI_SCORE', 'data', mode, 'to_label_{}.txt'.format(mode)) # 'roberta_mnli/self_logic/MNLI_SCORE/data/{}/to_label_{}.txt'.format(mode, mode)
    dev_matched_tsv = os.path.join('roberta_mnli', 'self_logic', 'MNLI_SCORE', 'data', mode, 'dev_matched.tsv'.format(mode))  #'roberta_mnli/examples/data/self_logic/{}/dev_matched.tsv'.format(mode)
    dev_mismatched_tsv = os.path.join('roberta_mnli', 'self_logic', 'MNLI_SCORE', 'data', mode, 'dev_mismatched.tsv'.format(mode)) 
    samples_json = os.path.join('roberta_mnli', 'self_logic', 'MNLI_SCORE', 'data', mode, 'samples.json') #'roberta_mnli/self_logic/MNLI_SCORE/data/{}/samples.json'.format(mode)
    intermepath = os.path.join('roberta_mnli', 'self_logic', 'MNLI_SCORE', 'data', mode) #'roberta_mnli/examples/data/self_logic/{}'.format(mode)
    score_file = os.path.join('roberta_mnli', 'self_logic', 'MNLI_SCORE', 'data', mode, 'mnli_{}.pred'.format(mode)) #'self_logic/MNLI_SCORE/data/{}/mnli.pred'.format(mode)
    jason_out = os.path.join('roberta_mnli', 'self_logic', 'MNLI_SCORE', 'data', mode, 'system_score_{}.json'.format(mode)) #'self_logic/labeled/system_score_baseline.json'

    file_form(sentences_pre, sentences, intermef)

    # to make paired sentences for MNLI.
    # Output 'examples/data/self_logic/{}/dev_matched.tsv'.format(mode), and 'self_logic/MNLI_SCORE/data/{}/samples.json'.format(mode)
    make_wordnet_para_intf(intermef, dev_matched_tsv, samples_json)

    # run MNLI. Output 'self_logic/MNLI_SCORE/data/{}/mnli.pred'.format(mode)
    shutil.copyfile(dev_matched_tsv, dev_mismatched_tsv)
    commandf = os.path.join('roberta_mnli', 'examples', 'run_glue.py')
    os.system('python '+commandf+' --model_type roberta --model_name_or_path roberta-large-mnli --task_name MNLI --do_eval --do_lower_case --data_dir '+intermepath+' --max_seq_length 128 --per_gpu_eval_batch_size=8 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir roberta_test1 --print_result --overwrite_cache')

    # compute score
    MNLI_scores = compute_score_intf(samples_json, score_file)

    with open(jason_out, 'w', encoding='utf8') as f:
        json.dump(MNLI_scores, f)

    return MNLI_scores