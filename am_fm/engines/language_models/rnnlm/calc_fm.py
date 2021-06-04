import argparse
import logging
import string
import codecs
import numpy as np
from tqdm import tqdm
import json


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--hyp_file", type=str, help="path to hypothesis file")
parser.add_argument("--ref_file", type=str, help="path to reference file")
parser.add_argument('--output_file', type=str)
args = parser.parse_args()


def calc_fm_batch(hyp_list, ref_list):
    per_sys_score = []
    for hyp in hyp_list:
        temp = []
        for ref in ref_list:
            temp.append(calc_fm(hyp, ref))
        #per_sys_score.append(np.amax(temp)-np.amin(temp))
        per_sys_score.append(np.amax(temp))
#        per_sys_score.append(np.amin(temp))
#        per_sys_score.append(np.mean(temp))
    return per_sys_score

def calc_fm(hyp, ref):
    return min(1/hyp, 1/ref)/max(1/hyp, 1/ref)

if __name__=='__main__':

    logging.info("Reading hypothesis perplexity -------------------------------------------------------")
    hyp_sent_ppl = []
    with codecs.open(args.hyp_file, mode='r', encoding='utf-8') as rf:
        for line in rf.readlines():
            hyp_sent_ppl.append(float(line.strip()))

    logging.info("Reading references perplexity -------------------------------------------------------")
    ref_sent_ppl = []
    with codecs.open(args.ref_file, mode='r', encoding='utf-8') as rf:
        for line in rf.readlines():
            ref_sent_ppl.append(float(line.strip()))

    logging.info("rearranging test cases -------------------------------------------------------------")
    hyp_per_all_sys = []
    hyp_per_sys = []
    for i, line in enumerate(hyp_sent_ppl):
        hyp_per_sys.append(line)
        #if (i + 1) % 2000 == 0:
        #    hyp_per_all_sys.append(hyp_per_sys)
        #    hyp_per_sys = []
    hyp_per_all_sys.append(hyp_per_sys)
    
    hyp_per_dialogues = []
    hyp_per_single_dialogue =[]
    for i, item in enumerate(hyp_per_all_sys[0]):
        hyp_per_single_dialogue.append(item)
        '''hyp_per_single_dialogue.append(hyp_per_all_sys[1][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[2][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[3][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[4][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[5][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[6][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[7][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[8][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[9][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[10][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[11][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[12][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[13][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[14][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[15][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[16][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[17][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[18][i])
        hyp_per_single_dialogue.append(hyp_per_all_sys[19][i])
        '''
        hyp_per_dialogues.append(hyp_per_single_dialogue)
        hyp_per_single_dialogue = []

    #assert len(hyp_per_dialogues) == 2000, 'number of hypothesis test cases not equal to 2000'

    ref_per_all_sys = []
    ref_per_sys = []
    for i, line in enumerate(ref_sent_ppl):
        ref_per_sys.append(line)
        #if (i + 1) % 2000 == 0:
        #    ref_per_all_sys.append(ref_per_sys)
        #    ref_per_sys = []
    ref_per_all_sys.append(ref_per_sys)

    ref_per_dialogues = []
    ref_per_single_dialogue =[]
    for i, item in enumerate(ref_per_all_sys[0]):
        ref_per_single_dialogue.append(item)
        '''
        ref_per_single_dialogue.append(ref_per_all_sys[1][i])
        ref_per_single_dialogue.append(ref_per_all_sys[2][i])
        ref_per_single_dialogue.append(ref_per_all_sys[3][i])
        ref_per_single_dialogue.append(ref_per_all_sys[4][i])
        ref_per_single_dialogue.append(ref_per_all_sys[5][i])
        ref_per_single_dialogue.append(ref_per_all_sys[6][i])
        ref_per_single_dialogue.append(ref_per_all_sys[7][i])
        ref_per_single_dialogue.append(ref_per_all_sys[8][i])
        ref_per_single_dialogue.append(ref_per_all_sys[9][i])
        ref_per_single_dialogue.append(ref_per_all_sys[10][i])
        '''
        ref_per_dialogues.append(ref_per_single_dialogue)
        ref_per_single_dialogue = []

    #assert len(ref_per_dialogues) == 2000, 'number of references test cases not equal to 2000'
    logging.info("Done rearranging test cases --------------------------------------------------------")

    # calculate FM score
    full_scores = []
    for hyp, ref in zip(hyp_per_dialogues, ref_per_dialogues):
        scores = calc_fm_batch(hyp, ref)
        #print(scores)
        full_scores.append(scores)

    with open(args.output_file, 'w') as f:
        json.dump({
            'scores': full_scores}, f)
    
    #full_scores = np.array(full_scores)
    #system_level_scores = np.mean(full_scores, axis=0)
    #logging.info('The final system level scores:')
    #for score in system_level_scores.tolist():
    #    print(score)

    
    
