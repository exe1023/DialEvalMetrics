import argparse
import logging
import string
import codecs
import numpy as np
from tqdm import tqdm
from lm import ArpaLM
import os
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--hyp_file", type=str, help="path to hypothesis file")
parser.add_argument("--ref_file", type=str, help="path to reference file")
args = parser.parse_args()

class calcScoresAMFM:
    def __init__(self, cfg):
        # Load configuration variables for language
        self.NGRAM_ORDER = cfg.SELECTED_NGRAM_ORDER
        self.fm = cfg.fm
        self.table = cfg.table
        self.models_dir = cfg.models_dir

    def load_models(self, prefix):
        # Check that the LM model exists
        lm_model = self.models_dir + '/' + prefix +  '.'  + str(self.NGRAM_ORDER) + '.lm'
        if not os.path.exists(lm_model):
            logging.info("******* ERROR: LM file " + lm_model + ' does not exists.')
            exit(-1)
        elif os.path.getsize(lm_model) == 0:
            logging.info("******* ERROR: LM file " + lm_model + ' is empty.')
            exit(-1)
        logging.info('Loading FM model...')
        self.lm = ArpaLM(lm_model)

    def calc_fm_batch(self, hyp_list, ref_list):
        per_sys_score = []
        for hyp in hyp_list:
            temp = []
            for ref in ref_list:
                temp.append(self.calculateFMMetric(ref, hyp))
            per_sys_score.append(np.amax(temp) - np.amin(temp))
        return per_sys_score

    # Function to calculate the FM metric using language models
    def calculateFMMetric(self, ref, tst):
        sent = '<s> ' + ref.strip() + ' </s>'
        aWords = sent.split()
        num_words_ref = len(aWords) - 2
        prob_ref = 0.0
        # Calculates the log-prob for the different n-grams
        for i in range(1, len(aWords)):
            prob_ref += self.lm.score(tuple(aWords[max(0, i-self.NGRAM_ORDER+1):i+1]))

        sent = '<s> ' + tst.strip() + ' </s>'
        aWords = sent.split()
        num_words_tst = len(aWords) - 2
        prob_tst = 0.0
        # Calculates the log-prob for the different n-grams
        for i in range(1, len(aWords)):
            prob_tst += self.lm.score(tuple(aWords[max(0, i-self.NGRAM_ORDER+1):i+1]))

        # Calculate the scaled probability
        prob_ref = np.exp(prob_ref / num_words_ref)
        prob_tst = np.exp(prob_tst / num_words_tst)
        return 1.0 - ((max(prob_tst, prob_ref) - min(prob_tst, prob_ref))/max(prob_tst, prob_ref))

    # # Pre-Processing for each sentence. In the case of languages different to English we perform tokenization
    # # per character
    # def preProcess(self, s):
    #     if len(s) == 0:  # To avoid empty lines
    #         return '_EMPTY_'

    #     # Remove some punctuation
    #     s = s.translate(self.table)
            
    #     tokens = s.split()

    #     new_tokens = []
    #     for token in tokens:
    #         if token.startswith('@'):
    #             new_tokens.append('<USER>')
    #         elif token.startswith('#'):
    #             new_tokens.append('<HASHTAG>')
    #         else:
    #             new_tokens.append(token)

    #     s = ' '.join(new_tokens).lower()
    #     return s

    # def doProcessFromStrings(self, ref, pred):
    #     ref = [self.preProcess(r) for r in ref]
    #     pred = [self.preProcess(p) for p in pred]
    #     return ref, pred

    # Evaluate the generated file. This file contains the reference (S_REF) and hypothesis (S_HYP)
    def processSubmission(self):
        logging.info('\n********* START PROCESSING SUBMISSION ************')
        try:
            target_sentences = []
            submission_sentences = []
            with codecs.open(args.hyp_file, mode='r', encoding='utf-8') as rf:
                for line in rf.readlines():
                    submission_sentences.append(line.strip())

            with codecs.open(args.ref_file, mode='r', encoding='utf-8') as rf:
                for line in rf.readlines():
                    target_sentences.append(line.strip())

            hyp_per_all_sys = []
            hyp_per_sys = []
            for i, line in enumerate(submission_sentences):
                hyp_per_sys.append(line)
                if (i + 1) % 2000 == 0:
                    hyp_per_all_sys.append(hyp_per_sys)
                    hyp_per_sys = []
            
            hyp_per_dialogues = []
            hyp_per_single_dialogue =[]
            for i, item in enumerate(hyp_per_all_sys[0]):
                hyp_per_single_dialogue.append(item)
                hyp_per_single_dialogue.append(hyp_per_all_sys[1][i])
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
                hyp_per_dialogues.append(hyp_per_single_dialogue)
                hyp_per_single_dialogue = []

            assert len(hyp_per_dialogues) == 2000, 'number of hypothesis test cases not equal to 2000'

            ref_per_all_sys = []
            ref_per_sys = []
            for i, line in enumerate(target_sentences):
                ref_per_sys.append(line)
                if (i + 1) % 2000 == 0:
                    ref_per_all_sys.append(ref_per_sys)
                    ref_per_sys = []

            ref_per_dialogues = []
            ref_per_single_dialogue =[]
            for i, item in enumerate(ref_per_all_sys[0]):
                ref_per_single_dialogue.append(item)
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
                ref_per_dialogues.append(ref_per_single_dialogue)
                ref_per_single_dialogue = []

            assert len(ref_per_dialogues) == 2000, 'number of references test cases not equal to 2000'


            # Calculate for each submitted sentence, given the reference, the AM, FM and combined scores
            results = []
            t = tqdm(total=2000)
            for num, (target, submission) in enumerate(zip(ref_per_dialogues, hyp_per_dialogues)):
                # (target, submission) = self.doProcessFromStrings(ref=target, pred=submission)

                res_fm = -1.0
                if self.fm is True:
                    res_fm = self.calc_fm_batch(submission, target)
                results.append((res_fm))
                t.update(1)
            t.close()
            full_scores = np.array(results)
            system_level_scores = np.mean(full_scores, axis=0)
            logging.info('The final system level scores:')
            for score in system_level_scores.tolist():
                logging.info(score)

            logging.info('********* END PROCESSING SUBMISSION ************\n')
        except Exception as e:
            logging.info(e)
            logging.info ('ERROR: Skipping submissions %s' % (args.hyp_file))

    
def main():
    import configuration as cfg

    cS = calcScoresAMFM(cfg)
    cS.load_models('twitter_full')
    cS.processSubmission()


if __name__ == '__main__':
    main()
    
    
