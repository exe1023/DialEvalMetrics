import os
import string

NFOLDS = 1  # Splits for the training data
MIN_SENTENCE_LENGTH = 5
MAX_SENTENCE_LENGTH = 40
MIN_SENTENCE_LENGTH_CHARS = 10
MAX_SENTENCE_LENGTH_CHARS = 80
NUM_MAX_CORES = 7
MIN_COUNTS = 1  # Number of times a word must occur to be included in the SVM model
MIN_NGRAM_ORDER = 1
MAX_NGRAM_ORDER = 5
SELECTED_NGRAM_ORDER = 5
MIN_LOG_PROB = -10
STARTING_VALUE_FEATURES = 0
FULL_AM_SIZE = 2500
OPT_AM_SIZE = 100
NUM_TRAIN_SENTS = 10000
ALPHA = 0.5

dictSizesTrain = {NUM_TRAIN_SENTS: FULL_AM_SIZE}  # Number of lines to use for training the SVD + final SVD dimension
num_cores_mono = {FULL_AM_SIZE: 5}
lang_char_tokenization = []  # Languages that require character tokenization instead of word-based
type_vectorizer = 'counts'
overwrite_all = False  # Whether if re-do all the matrices and process by overwriting existing files
overwrite_svd = False
bAverageFiles = False  # Flag to indicate if we want to average results from different SVD matrices
am = True
fm = True
lang = 'en'

sc = set(['-', "'", '%', '#', '@', '<', '>'])
to_remove = ''.join([c for c in string.punctuation if c not in sc])
table = dict((ord(char), u'') for char in to_remove)


root_dir = os.path.dirname(os.path.realpath(__file__))
submissions_dir = root_dir + '/'
train_data_dir = root_dir + '/'
scripts_dir = root_dir + '/tools/'
dir_lm_out = root_dir + '/lms/'
scores_dir = root_dir + '/scores/'
models_dir = root_dir + '/models'

# type_vectorizer = 'tfidf'
dir_svd_mono = root_dir + '/svd_mono_' + type_vectorizer + '/'
dir_svd_cross = root_dir + '/svd_cross_' + type_vectorizer
aTypeLingualExp = ['mono']
monoSizesSVD = dict()
crossSizesSVD = dict()
aSizesTrain = [NUM_TRAIN_SENTS]
monoSizesSVD[NUM_TRAIN_SENTS] = [100, 200, 300, 500, 1000, 2000, 2500]  # The last one must be always the maximum value of the model
crossSizesSVD[NUM_TRAIN_SENTS] = [50, 100, 150, 250]


# Relative path for files to be used for training the language models (they can be different to the training data for
# the system (however in most cases will be the same). Each entry is processed separately
filesPerLanguageForLM = {
    # 'usr': [
    #     'twitter/train',
    #     # 'opensubtitles/train'
    # ],
    'sys': [
        'twitter/train',
        # 'opensubtitles/train'
    ],

}

# Files used for training the MT system. There most be files X.usr.txt and X.sys.txt
filesPerLanguage = {
    'usr-sys': [
        'twitter/train',
        # 'opensubtitles/train'
    ],
}

# Files used to evaluate the performance of a given system
# submissionsPerLanguagePerTask = {
#     'usr-sys': {
#         'twitter': {
#            'source': 'twitter/eval.usr',  # The source parallel file
#            'reference': 'twitter/eval.sys',  # The target reference
#            'test_id': 'twitter_eval',  # An identifier for the task
#            'submissions': [  # The actual prediction of different systems for the target language
#                'twitter/baseline.eval.sys',
#            ],
#         },
#     },
# }

submissionsPerLanguagePerTask = {
    'usr-sys': {
        'twitter': {
           'source': 'twitter/eval.usr',  # The source parallel file
           'reference': 'dstc6_t2_evaluation/references/original_refs.txt',  # The target reference
           'test_id': 'twitter_eval',  # An identifier for the task
           'submissions': [  # The actual prediction of different systems for the target language
               'dstc6_t2_evaluation/hypotheses/S_1.txt',
               'dstc6_t2_evaluation/hypotheses/S_2.txt',
               'dstc6_t2_evaluation/hypotheses/S_3.txt',
               'dstc6_t2_evaluation/hypotheses/S_4.txt',
               'dstc6_t2_evaluation/hypotheses/S_5.txt',
               'dstc6_t2_evaluation/hypotheses/S_6.txt',
               'dstc6_t2_evaluation/hypotheses/S_7.txt',
               'dstc6_t2_evaluation/hypotheses/S_8.txt',
               'dstc6_t2_evaluation/hypotheses/S_9.txt',
               'dstc6_t2_evaluation/hypotheses/S_10.txt',
               'dstc6_t2_evaluation/hypotheses/S_11.txt',
               'dstc6_t2_evaluation/hypotheses/S_12.txt',
               'dstc6_t2_evaluation/hypotheses/S_13.txt',
               'dstc6_t2_evaluation/hypotheses/S_14.txt',
               'dstc6_t2_evaluation/hypotheses/S_15.txt',
               'dstc6_t2_evaluation/hypotheses/S_16.txt',
               'dstc6_t2_evaluation/hypotheses/S_17.txt',
               'dstc6_t2_evaluation/hypotheses/S_18.txt',
               'dstc6_t2_evaluation/hypotheses/S_19.txt',
               'dstc6_t2_evaluation/hypotheses/S_20.txt',
           ],
        },
    },
}
