import argparse
import logging
import string
import numpy as np
from numpy import linalg as LA
import jsonlines
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument("--hyp_file", type=str, help="path to hypothesis file")
parser.add_argument("--ref_file", type=str, help="path to reference file")
parser.add_argument("--strategy", type=str, 
     help="am score computation strategy", default='top-layer-embedding-average')
args = parser.parse_args()


def calc_am_batch(hyp_list, ref_list):
    score_mat = cosine_similarity(np.array(hyp_list), np.array(ref_list))
    score_mat = np.amax(score_mat, axis=1).T
    return score_mat

def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)
    

if __name__=='__main__':

    logging.info("Loading hypothesis features -------------------------------------------------------")
    hyp_sent_embedding = []
    tq = tqdm(total=40000)
    with jsonlines.open(args.hyp_file) as reader:
        for obj in reader:
            if args.strategy == 'top-layer-embedding-average':
                # embedding average
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.sum(obj_emb, axis=0)
                obj_emb =  obj_emb / LA.norm(obj_emb)
            if args.strategy == 'all-layer-embedding-average':
                # embedding average
                obj_emb = np.array([token['layers'][0]['values'] + token['layers'][1]['values'] \
                 + token['layers'][2]['values'] + token['layers'][3]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.sum(obj_emb, axis=0)
                obj_emb =  obj_emb / LA.norm(obj_emb)
            elif args.strategy == 'top-layer-max-pool':
                # top layer max pooling
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'top-layer-mean-pool':
                # top layer mean pooling
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.mean(obj_emb, axis=0)
            elif args.strategy == 'second-layer-max-pool':
                # second layer max pooling
                obj_emb = np.array([token['layers'][1]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'all-layer-concat-mean-pooling':
                # all-layer-concat-mean-pooling
                obj_emb = np.array([token['layers'][0]['values'] + token['layers'][1]['values'] \
                 + token['layers'][2]['values'] + token['layers'][3]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.mean(obj_emb, axis=0)
            elif args.strategy == 'all-layer-concat-max-pooling':
                # all-layer-concat-mean-pooling
                obj_emb = np.array([token['layers'][0]['values'] + token['layers'][1]['values'] \
                 + token['layers'][2]['values'] + token['layers'][3]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'all-layer-mean-max-pooling':
                # all-layer-mean-max-pooling
                obj_emb = []
                for token in obj['features']:
                    temp = np.mean(np.array([token['layers'][0]['values'], token['layers'][1]['values'], 
                    token['layers'][2]['values'], token['layers'][3]['values']], dtype='float32'), axis=0)
                    obj_emb.append(temp)
                obj_emb = np.array(obj_emb)
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'all-layer-max-max-pooling':
                # all-layer-mean-max-pooling
                obj_emb = []
                for token in obj['features']:
                    temp = np.max(np.array([token['layers'][0]['values'], token['layers'][1]['values'], 
                    token['layers'][2]['values'], token['layers'][3]['values']], dtype='float32'), axis=0)
                    obj_emb.append(temp)
                obj_emb = np.array(obj_emb)
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'all-layer-concat-vector-extrema':
                # all-layer-concat-max-pooling
                obj_emb = np.array([token['layers'][0]['values'] + token['layers'][1]['values'] + 
                    token['layers'][2]['values'] + token['layers'][3]['values'] for token in obj['features']], dtype='float32')
                obj_emb = absmaxND(obj_emb, axis=0)
            elif args.strategy == 'top-layer-concat-vector-extrema':
                 # top-layer-vector-extrema
                 obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                 obj_emb = absmaxND(obj_emb, axis=0)
            elif args.strategy == 'all-layer-addition-max-pooling':
                 # all-layer-addition-max-pooling
                 obj_emb = []
                 for token in obj['features']:
                     temp = np.array([token['layers'][0]['values'], token['layers'][1]['values'], 
                     token['layers'][2]['values'], token['layers'][3]['values']], dtype='float32')
                     temp = np.sum(temp, axis=0)
                     obj_emb.append(temp)
                 obj_emb = np.array(obj_emb)
                 obj_emb = np.max(obj_emb, axis=0)       
            hyp_sent_embedding.append(obj_emb)
            tq.update(1)
    tq.close()
    logging.info("Done loading hypothesis features ---------------------------------------------------")


    logging.info("Loading references features --------------------------------------------------------")
    ref_sent_embedding = []
    tq = tqdm(total=22000)
    with jsonlines.open(args.ref_file) as reader:
        for obj in reader:
            if args.strategy == 'top-layer-embedding-average':
                # embedding average
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.sum(obj_emb, axis=0)
                obj_emb =  obj_emb / LA.norm(obj_emb)
            if args.strategy == 'all-layer-embedding-average':
                # embedding average
                obj_emb = np.array([token['layers'][0]['values'] + token['layers'][1]['values'] \
                 + token['layers'][2]['values'] + token['layers'][3]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.sum(obj_emb, axis=0)
                obj_emb =  obj_emb / LA.norm(obj_emb)
            elif args.strategy == 'top-layer-max-pool':
                # top layer max pooling
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'top-layer-mean-pool':
                # top layer mean pooling
                obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.mean(obj_emb, axis=0)
            elif args.strategy == 'second-layer-max-pool':
                # second layer max pooling
                obj_emb = np.array([token['layers'][1]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'all-layer-concat-mean-pooling':
                # all-layer-concat-mean-pooling
                obj_emb = np.array([token['layers'][0]['values'] + token['layers'][1]['values'] \
                 + token['layers'][2]['values'] + token['layers'][3]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.mean(obj_emb, axis=0)
            elif args.strategy == 'all-layer-concat-max-pooling':
                # all-layer-concat-mean-pooling
                obj_emb = np.array([token['layers'][0]['values'] + token['layers'][1]['values'] \
                 + token['layers'][2]['values'] + token['layers'][3]['values'] for token in obj['features']], dtype='float32')
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'all-layer-mean-max-pooling':
                # all-layer-mean-max-pooling
                obj_emb = []
                for token in obj['features']:
                    temp = np.mean(np.array([token['layers'][0]['values'], token['layers'][1]['values'], 
                    token['layers'][2]['values'], token['layers'][3]['values']], dtype='float32'), axis=0)
                    obj_emb.append(temp)
                obj_emb = np.array(obj_emb)
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'all-layer-max-max-pooling':
                # all-layer-mean-max-pooling
                obj_emb = []
                for token in obj['features']:
                    temp = np.max(np.array([token['layers'][0]['values'], token['layers'][1]['values'], 
                    token['layers'][2]['values'], token['layers'][3]['values']], dtype='float32'), axis=0)
                    obj_emb.append(temp)
                obj_emb = np.array(obj_emb)
                obj_emb = np.max(obj_emb, axis=0)
            elif args.strategy == 'all-layer-concat-vector-extrema':
                # all-layer-concat-max-pooling
                obj_emb = np.array([token['layers'][0]['values'] + token['layers'][1]['values'] + 
                    token['layers'][2]['values'] + token['layers'][3]['values'] for token in obj['features']], dtype='float32')
                obj_emb = absmaxND(obj_emb, axis=0)
            elif args.strategy == 'top-layer-concat-vector-extrema':
                 # top-layer-vector-extrema
                 obj_emb = np.array([token['layers'][0]['values'] for token in obj['features']], dtype='float32')
                 obj_emb = absmaxND(obj_emb, axis=0)
            elif args.strategy == 'all-layer-addition-max-pooling':
                 # all-layer-addition-max-pooling
                 obj_emb = []
                 for token in obj['features']:
                     temp = np.array([token['layers'][0]['values'], token['layers'][1]['values'], 
                     token['layers'][2]['values'], token['layers'][3]['values']], dtype='float32')
                     temp = np.sum(temp, axis=0)
                     obj_emb.append(temp)
                 obj_emb = np.array(obj_emb)
                 obj_emb = np.max(obj_emb, axis=0)  
            ref_sent_embedding.append(obj_emb)
            tq.update(1)
    tq.close()
    logging.info("Done loading references features ---------------------------------------------------")


    logging.info("rearranging test cases -------------------------------------------------------------")
    hyp_per_all_sys = []
    hyp_per_sys = []
    for i, line in enumerate(tqdm(hyp_sent_embedding)):
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
    for i, line in enumerate(tqdm(ref_sent_embedding)):
        ref_per_sys.append(line)
        #if (i + 1) % 2000 == 0:
        #    ref_per_all_sys.append(ref_per_sys)
        #    ref_per_sys = []
    ref_per_all_sys.append(ref_per_sys)

    ref_per_dialogues = []
    ref_per_single_dialogue =[]
    for i, item in enumerate(ref_per_all_sys[0]):
        ref_per_single_dialogue.append(item)
        '''ref_per_single_dialogue.append(ref_per_all_sys[1][i])
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

    # calculate AM score
    full_scores = []
    for hyp, ref in zip(hyp_per_dialogues, ref_per_dialogues):
        scores = calc_am_batch(hyp, ref)
        full_scores.append(scores)
    
    #print(full_scores)
    full_scores = np.array(full_scores)
    system_level_scores = np.mean(full_scores, axis=0)
    logging.info('The final system level scores:')
    for score in system_level_scores.tolist():
        print(score)
    
    with open(args.output_dir + '/am_scores.json', 'w') as f:
        json.dump(
            {'scores': full_scores.tolist()}, f
        )

    
    
