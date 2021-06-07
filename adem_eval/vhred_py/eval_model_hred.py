"""
Dialogue Evaluation Model using VHRED

This code learns to predict human scores
using a linear model on top of VHRED embeddings.
"""

import csv
import imp
import math
import os
import pickle
import sys
import time
from random import randint

import numpy as np
import scipy
import theano
import theano.tensor as T
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from .vhred_compute_dialogue_embeddings import compute_encodings
from .vhred_dialog_encdec import DialogEncoderDecoder
from .vhred_state import prototype_state

os.sys.path.insert(0, '../TwitterData/BPE/subword_nmt')

from apply_bpe import BPE

imp.reload(sys)
sys.setdefaultencoding('utf-8')

import random

random.seed(0)


def load_data(filein):
    # Input: csv file name (string)
    # Output: 
    with open(filein, 'r') as f1:
        data = []
        csv1 = csv.reader(f1)
        for row in csv1:
            data.append(row)
    return data


def get_score(data):
    score = []
    for row in data:
        score.append(float(row[-1]))
    return score


def preprocess_tweet(s):
    s = s.replace('@user', '<at>').replace('&lt;heart&gt;', '<heart>').replace('&lt;number&gt;', '<number>').replace(
        '  ', ' </s> ').replace('  ', ' ')
    # Make sure we end with </s> token
    while s[-1] == ' ':
        s = s[0:-1]
    if not s[-5:] == ' </s>':
        s = s + ' </s>'

    return s


def strs_to_idxs(data, bpe, str_to_idx):
    ''' Encodes strings in BPE form '''
    out = []
    for row in data:
        bpe_segmented = bpe.segment(row.strip())
        # Note: there shouldn't be any unknown tokens with BPE!
        # out.append([str_to_idx[word] for word in bpe_segmented.split()])
        out.append([str_to_idx[word] for word in bpe_segmented.split() if word in str_to_idx])

    return out


def idxs_to_strs(data, bpe, idx_to_str):
    ''' Converts from BPE form to strings '''
    out = []
    for row in data:
        out.append(' '.join([idx_to_str[idx] for idx in row if idx in idx_to_str]).replace('@@ ', ''))
    return out


def get_context(data):
    out = []
    for row in data:
        out.append('</s> ' + preprocess_tweet(row[0][5:-2]))
        # TODO: this also puts the </s> token at the beginning of every context... 
        # is that what we want?

    return out


def get_gtresponse(data):
    out = []
    for row in data:
        out.append(preprocess_tweet(row[2][5:-2]))

    return out


def get_modelresponse(data):
    out = []
    for row in data:
        out.append(preprocess_tweet(row[1][5:-2]))
    return out


def get_twitter_data(clean_data_file, context_file, gt_file):
    '''
    Loads Twitter data from dictionaries.
    '''
    with open(clean_data_file, 'rb') as f1:
        clean_data = pickle.load(f1)
    with open(context_file, 'rb') as f1:
        contexts = pickle.load(f1)
    gt_unordered = []
    with open(gt_file, 'r') as f1:
        for row in f1:
            gt_unordered.append(row)

    # Retrieve scores and valid context ids from clean_data.pkl
    valid_contextids = []
    score_dic = {}
    for user in clean_data:
        for dic in clean_data[user]:
            if int(dic['c_id']) >= 0:
                score_dic[dic['c_id']] = [dic['overall1'], dic['overall2'], dic['overall3'], dic['overall4']]
                # NOTE: although there's some contexts with multiple responses, the above line basically uses the last score
                # for each context (so there is no context overlap between train and test)
    context_list = []
    gtresponses = []
    model_responses = []
    scores = []

    # Retrieve contexts and model responses from contexts.pkl
    for c in contexts:
        if int(c[0]) in score_dic:
            valid_contextids.append(int(c[0]))
            context_list.append(c[1])
            model_responses.append(c[2:6])
            scores.append(score_dic[int(c[0])])
            gtresponses.append(gt_unordered[int(c[0])])
    model_responses = [i for sublist in model_responses for i in sublist]  # flatten list
    scores = [float(i) for sublist in scores for i in sublist]  # flatten list

    valid_contextids.sort()
    return context_list, gtresponses, model_responses, scores, valid_contextids


def combine_contextids(cid1, cid2):
    # Combines 2 lists of context ids
    for cid in cid2:
        if cid not in cid1:
            cid1.append(cid)
    cid1.sort()
    return cid1


def flatten(l1):
    return [i for sublist in l1 for i in sublist]


def construct_filter(model_map, filtname, name):
    filter_list = []
    model_list = []
    for key in model_map:
        model_list.append(model_map[key])
    model_list = flatten(model_list)
    for model in model_list:
        if model == filtname:
            filter_list.append(0)
        else:
            filter_list.append(1)
    if name == 'train':
        return filter_list
    else:
        return [1 - i for i in filter_list]


def apply_train_filter(train_x, train_y, train_feat, train_filt):
    """
    Applies filter to the training data, if you want to remove a certain model
    """
    train_x_l = list(train_x);
    train_y_l = list(train_y);
    train_feat_l = list(train_feat)
    train_x_l2 = [];
    train_y_l2 = [];
    train_feat_l2 = []

    for i in range(len(train_filt)):
        if train_filt[i] == 1:
            train_x_l2.append(train_x_l[i])
            train_y_l2.append(train_y_l[i])
            train_feat_l2.append(train_feat_l[i])

    return np.array(train_x_l2), np.array(train_y_l2), np.array(train_feat_l2)


def apply_test_filter(val_x, val_y, val_feat, test_x, test_y, test_feat, val_filt, test_filt):
    """
    Applies filter to the val/test data, if you want to remove certain models
    """
    val_x_l = list(val_x);
    val_y_l = list(val_y);
    val_feat_l = list(val_feat)
    val_x_l2 = [];
    val_y_l2 = [];
    val_feat_l2 = []
    test_x_l = list(test_x);
    test_y_l = list(test_y);
    test_feat_l = list(test_feat)
    test_x_l2 = [];
    test_y_l2 = [];
    test_feat_l2 = []

    for i in range(len(val_filt)):
        if val_filt[i] == 1:
            val_x_l2.append(val_x_l[i])
            val_y_l2.append(val_y_l[i])
            val_feat_l2.append(val_feat_l[i])

    for i in range(len(test_filt)):
        if test_filt[i] == 1:
            test_x_l2.append(test_x_l[i])
            test_y_l2.append(test_y_l[i])
            test_feat_l2.append(test_feat_l[i])

    return np.array(val_x_l2), np.array(val_y_l2), np.array(val_feat_l2), \
           np.array(test_x_l2), np.array(test_y_l2), np.array(test_feat_l2)


def filter_modelmap(model_map, valid_contextids):
    for key in list(model_map.keys()):
        if key not in valid_contextids:
            del model_map[key]
    return model_map


def fixmodelmap2(model_map2):
    for key in list(model_map2.keys()):
        if key <= 988:
            del model_map2[key]
    return model_map2


# Compute mean and range of dot products to find right constants in model
def compute_init_values(emb):
    prod_list = []
    for i in range(len(emb[0][0])):
        prod_list.append(np.dot(emb[i, 0], emb[i, 2]) + np.dot(emb[i, 1], emb[i, 2]))
    return sum(prod_list) / float(len(prod_list)), max(prod_list) - min(prod_list)


# Prints BLEU, METEOR, etc. correlation scores on the test set
def show_overlap_scores(twitter_gtresponses, twitter_modelresponses, twitter_human_scores, test_pct, contexts,
                        liu=False):
    # Align ground truth with model responses
    temp_gt = []
    if not liu:
        twitter_gtresponses = extend_by_four(twitter_gtresponses)
        contexts = extend_by_four(contexts)

    assert len(twitter_modelresponses) == len(twitter_gtresponses)
    assert len(twitter_modelresponses) == len(twitter_human_scores)

    # test_index = 0 # If you want to evaluate on the whole dataset
    test_index = int((1 - test_pct) * len(twitter_modelresponses))
    test_gtresponses = twitter_gtresponses[test_index:]
    test_modelresponses = twitter_modelresponses[test_index:]
    test_scores = twitter_human_scores[test_index:]

    bleu1_list = []
    bleu2_list = []
    bleu3_list = []
    bleu4_list = []
    rouge_list = []
    meteor_list = []
    # print Meteor()._score(test_gtresponses[0], test_modelresponses[0])
    start = time.time()
    for i in range(len(test_modelresponses)):
        dict_in1 = {0: [test_gtresponses[i].encode('utf-8').strip()]}
        dict_in2 = {0: [test_modelresponses[i].encode('utf-8').strip()]}
        bleu1_list.append(Bleu(1).compute_score(dict_in1, dict_in2)[0][0])
        bleu2_list.append(Bleu(2).compute_score(dict_in1, dict_in2)[0][1])
        bleu3_list.append(Bleu(3).compute_score(dict_in1, dict_in2)[0][2])
        bleu4_list.append(Bleu(4).compute_score(dict_in1, dict_in2)[0][3])
        rouge_list.append(Rouge().compute_score(dict_in1, dict_in2)[0])
        # meteor_list.append(Meteor().compute_score(dict_in1, dict_in2)[0]) # Can comment this out since it takes so long
        # if i%10 == 0:
        #    print str(i) + ' / ' + str(len(test_modelresponses))
        #    print time.time() - start
    print('Took: ' + str(time.time() - start))
    return bleu2_list, rouge_list


'''
    for i in xrange(len(twitter_gtresponses)):
        print '\n ' + ' + '*40
        print contexts[i]
        print twitter_gtresponses[i]
        print twitter_modelresponses[i]
        print bleu2_list[i]
        print rouge_list[i]
        print twitter_human_scores[i]

    test_y_gauss = []
    for y in test_scores:
        test_y_gauss.append(y + np.random.normal(0, 0.3))
    
    suffix = ''
    if test_index == 0:
        suffix = '_full'
    make_plot(bleu1_list, test_y_gauss, './results/word_overlap/bleu1_gauss' + suffix + '.png')
    make_plot(bleu2_list, test_y_gauss, './results/word_overlap/bleu2_gauss' + suffix + '.png')
    make_plot(bleu3_list, test_y_gauss, './results/word_overlap/bleu3_gauss' + suffix + '.png')
    make_plot(bleu4_list, test_y_gauss, './results/word_overlap/bleu4_gauss' + suffix + '.png')
    make_plot(rouge_list, test_y_gauss, './results/word_overlap/rouge_gauss' + suffix + '.png')
    #make_plot(meteor_list, test_y_gauss, './results/word_overlap/meteor_gauss.png')

    metric_list = [bleu1_list, bleu2_list, bleu3_list, bleu4_list, rouge_list, meteor_list]
    metric_name = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge', 'meteor']
    for metric, name in zip(metric_list, metric_name):
        spearman = scipy.stats.spearmanr(metric, test_scores)
        pearson = scipy.stats.pearsonr(metric, test_scores)
        print 'For ' + name + ' score:'
        print spearman
        print pearson
        with open('./metric_scores/' + name + '.txt', 'w') as f1:
            for score in metric:
                f1.write(str(score)+'\n')
'''


# Computes PCA decomposition for Liu et al.'s data (using PCA from train of original data)
def compute_liu_pca(pca_components, twitter_dialogue_embeddings, pca):
    tw_embeddings_pca = np.zeros((twitter_dialogue_embeddings.shape[0], 3, pca_components))
    for i in range(3):
        tw_embeddings_pca[:, i] = pca.transform(twitter_dialogue_embeddings[:, i])
    return tw_embeddings_pca


# Computes PCA decomposition for the context, gt responses, and model responses separately
def compute_separate_pca(pca_components, twitter_dialogue_embeddings):
    pca = PCA(n_components=pca_components)
    tw_embeddings_pca = np.zeros((twitter_dialogue_embeddings.shape[0], 3, pca_components))
    for i in range(3):
        tw_embeddings_pca[:, i] = pca.fit_transform(twitter_dialogue_embeddings[:, i])
    return tw_embeddings_pca


# Computes PCA decomposition for the context, gt responses, and model responses together
# NOTE: this computes the PCA on the training embeddings, and then applies them to the
# test embeddings (it does not compute PCA on the testing embeddings)
def compute_pca(pca_components, twitter_dialogue_embeddings, train_index, val_index):
    pca = PCA(n_components=pca_components)
    tw_nonpca_train = twitter_dialogue_embeddings[:train_index]
    tw_nonpca_val = twitter_dialogue_embeddings[train_index:val_index]
    tw_nonpca_test = twitter_dialogue_embeddings[val_index:]

    num_ex_train = tw_nonpca_train.shape[0]
    num_ex_test = tw_nonpca_test.shape[0]  # NOTE: we assume the val/ test set sizes are the same
    dim = twitter_dialogue_embeddings.shape[2]
    tw_embeddings_pca_train = np.zeros((num_ex_train * 3, dim))
    tw_embeddings_pca_val = np.zeros((num_ex_test * 3, dim))
    tw_embeddings_pca_test = np.zeros((num_ex_test * 3, dim))
    for i in range(3):
        tw_embeddings_pca_train[num_ex_train * i: num_ex_train * (i + 1), :] = tw_nonpca_train[:, i]
        tw_embeddings_pca_val[num_ex_test * i: num_ex_test * (i + 1), :] = tw_nonpca_val[:, i]
        tw_embeddings_pca_test[num_ex_test * i: num_ex_test * (i + 1), :] = tw_nonpca_test[:, i]
    tw_embeddings_pca_train = pca.fit_transform(tw_embeddings_pca_train)
    tw_embeddings_pca_val = pca.transform(tw_embeddings_pca_val)
    tw_embeddings_pca_test = pca.transform(tw_embeddings_pca_test)
    tw_emb_train = np.zeros((num_ex_train, 3, pca_components))
    tw_emb_val = np.zeros((num_ex_test, 3, pca_components))
    tw_emb_test = np.zeros((num_ex_test, 3, pca_components))
    for i in range(3):
        tw_emb_train[:, i] = tw_embeddings_pca_train[num_ex_train * i: num_ex_train * (i + 1), :]
        tw_emb_val[:, i] = tw_embeddings_pca_val[num_ex_test * i: num_ex_test * (i + 1), :]
        tw_emb_test[:, i] = tw_embeddings_pca_test[num_ex_test * i: num_ex_test * (i + 1), :]
    return tw_emb_train, tw_emb_val, tw_emb_test, pca


# Compute model embeddings for contexts or responses 
# Embedding type can be 'CONTEXT' or 'DECODER'
# NOTE: vhred_retrieval.py has code for saving the embeddings
# from the whole Twitter dataset into .pkl files
def compute_model_embeddings(data, model, embedding_type):
    model_compute_encoding = model.build_encoder_function()
    model_compute_decoder_encoding = model.build_decoder_encoding()
    model.bs = 20
    embeddings = []
    context_ids_batch = []
    batch_index = 0
    batch_total = int(math.ceil(float(len(data)) / float(model.bs)))
    counter = 0
    for context_ids in data:
        counter += 1
        context_ids_batch.append(context_ids)

        if len(context_ids_batch) == model.bs or counter == len(data):
            batch_index += 1
            # if counter == len(data):
            #    model.bs = counter % model.bs
            print('     Computing embeddings for batch ' + str(batch_index) + ' / ' + str(batch_total))
            encs = compute_encodings(context_ids_batch, model, model_compute_encoding, model_compute_decoder_encoding,
                                     embedding_type)
            for i in range(len(encs)):
                embeddings.append(encs[i])

            context_ids_batch = []

    return embeddings


def get_len_features(modelresponses, num_examples):
    l = []
    for i in range(num_examples):
        l.append(len(modelresponses[i]))
    return np.array(l)


def get_auxiliary_features(contexts, gtresponses, modelresponses, num_examples):
    aux_features = np.zeros((num_examples, 5))
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    meteor = []
    rouge = []
    for i in range(num_examples):
        bleu1.append(Bleu(1).compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0][0])
        bleu2.append(Bleu(2).compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0][1])
        bleu3.append(Bleu(3).compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0][2])
        bleu4.append(Bleu(4).compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0][3])
        rouge.append(Rouge().compute_score({0: [gtresponses[i]]}, {0: [modelresponses[i]]})[0])
    aux_features[:, 0] = bleu1
    aux_features[:, 1] = bleu2
    aux_features[:, 2] = bleu3
    aux_features[:, 3] = bleu4
    aux_features[:, 4] = rouge
    return aux_features


def clean_str(s):
    return s.replace('<br />', '').replace('<first_speaker>', 'A:').replace('<second_speaker>', 'A:').replace( \
        '<third_speaker>', 'A:').replace('\n', '')


def clean_list(l):
    l2 = []
    for s in l:
        l2.append(clean_str(s))
    return l2


def make_plot(model_scores, human_scores, filename):
    pp.clf()
    pp.plot(human_scores, model_scores, 'ko')
    pp.plot(np.unique(human_scores), np.poly1d(np.polyfit(human_scores, model_scores, 1))(np.unique(human_scores)), 'r',
            linewidth=2.5)
    pp.xlabel('Human scores', fontsize=19)
    pp.ylabel('Model scores', fontsize=19)
    pp.savefig(filename)


def make_line_plot(model_scores, human_scores, filename):
    pp.clf()
    pp.plot(human_scores, model_scores)
    pp.xlabel('Number of epochs')
    pp.savefig(filename)


def extend_by_four(l_in):
    l_temp = []
    for i in range(len(l_in)):
        l_temp.append([l_in[i]] * 4)
    return flatten(l_temp)


def calc_system_scores(model_map, predictions):
    # Mike ended up doing this, so disregard
    model_list = ['de', 'human', 'tfidf', 'hred']
    scores = []
    for model in model_list:
        filt = construct_filter(model_map, model, 'test')
        pred_list = []
        for i in range(len(pred)):
            if filt[i] == 1:
                pred_list.append(pred[i])
        scores.append(np.mean(np.array(pred_list)))


#####################
# Code for learning #
#####################

def set_shared_variable(x):
    return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)


def correlation(output, score):
    """
    Returns a list of the Spearman and Pearson ranks, and the significance
    """
    cor = []
    spearman = scipy.stats.spearmanr(output, score)
    pearson = scipy.stats.pearsonr(output, score)
    cor.append(spearman)
    cor.append(pearson)
    return cor


class LinearEvalModel(object):
    """
    Class for learning two weight matrices, M and N, and bias b
    Output is sigmoid( cMr + r'Nr )

    input has shape (batch size x 3 x emb dimensionality)
    """

    def __init__(self, input, feat, emb_dim, batch_size, init_mean, init_range, mean_length, feat_dim=0):
        # Matrices for predicting score
        mean_length = float(mean_length)
        self.emb_dim = emb_dim
        self.M = theano.shared(np.eye(emb_dim).astype(theano.config.floatX), borrow=True)
        self.N = theano.shared(np.eye(emb_dim).astype(theano.config.floatX), borrow=True)
        # Fair projection matrix 
        self.F = theano.shared(np.eye(emb_dim).astype(theano.config.floatX), borrow=True)
        # Parameters for auxiliary features
        self.f = theano.shared(np.zeros((feat_dim,)).astype(theano.config.floatX), borrow=True)

        # Set embeddings by slicing tensor
        self.emb_context = input[:, 0, :]
        self.emb_true_response = input[:, 1, :]
        self.emb_response = input[:, 2, :]
        self.feat = feat
        self.x = input

        # Compute score predictions
        self.pred1 = T.sum(self.emb_context * T.dot(self.emb_response, self.M), axis=1)
        self.pred2 = T.sum(self.emb_true_response * T.dot(self.emb_response, self.N), axis=1)
        # self.pred4 = T.sum(self.emb_response_fair * T.dot(self.emb_response, self.M), axis=1)
        # self.pred3 = T.dot(self.feat, self.f)
        self.pred = self.pred1 + self.pred2  # + self.pred3
        # self.pred = T.sum(T.dot(self.emb_response, self.f), axis=1) + 0*T.sum(self.feat) # If only using response
        self.output = 2.5 + 5 * (self.pred - init_mean) / init_range  # to re-scale dot product values to [0,5] range

        # Feed-forward neural net for predicting length
        # n_hid = emb_dim
        # self.q = theano.shared(np.zeros((emb_dim, n_hid)).astype(theano.config.floatX), borrow=True)
        # self.q2 = theano.shared(np.zeros((n_hid,)).astype(theano.config.floatX), borrow=True)
        # self.b = theano.shared(np.zeros((n_hid,)))
        # self.b2 = theano.shared(1. * mean_length)

        # l1 = T.dot(self.emb_response_fair, self.q) + self.b
        # l1 = T.nnet.relu(l1)
        # l2 = T.dot(l1, self.q2) + self.b2 * np.ones((batch_size,))
        # self.nuis = l2

        # Using Lasagne
        # net = lasagne.layers.InputLayer((None, emb_dim), self.emb_response_fair.T)
        # net = lasagne.layers.DenseLayer(net, num_units=n_hid)
        # net = lasagne.layers.DenseLayer(net, num_units=1, b=lasagne.init.Constant(mean_length),\
        #        nonlinearity=lasagne.nonlinearities.linear)
        # self.nuis = lasagne.layers.get_output(net)
        # self.net = net

    def squared_error(self, score):
        return T.mean((self.output - score) ** 2)

    def linear_error(self, score):
        return T.mean(T.log(T.exp(2 * (self.output - score)) + 1) - (self.output - score))

    def l2_regularization(self):
        return self.M.norm(2) + self.N.norm(2)

    def l1_regularization(self):
        return self.M.norm(1) + self.N.norm(1)

    def get_params(self):
        return [self.M, self.N]

    def set_params(self, param_list):
        self.M = param_list[0]
        self.N = param_list[1]
        return


def train(train_x, val_x, test_x, train_y, val_y, test_y, train_l, val_l, test_l, \
          init_mean, init_range, learning_rate=0.01, num_epochs=100, \
          batch_size=16, l2reg=0, l1reg=0, train_feat=None, val_feat=None, test_feat=None, pca_name=None, \
          exp_folder=None, test_contexts=None, test_modelresponses=None, test_gtresponses=None,
          bleu2_list=None, rouge_list=None, mean_length=None):
    compile_start = time.time()
    print('...building model')
    n_train_batches = train_x.shape[0] / batch_size
    emb_dim = int(train_x.shape[2])
    feat_dim = int(train_feat.shape[1])

    train_y_values = train_y
    val_y_values = val_y
    train_l_values = train_l
    train_x = set_shared_variable(train_x)
    val_x = set_shared_variable(val_x)
    test_x = set_shared_variable(test_x)
    train_y = set_shared_variable(train_y)
    train_feat = set_shared_variable(train_feat)
    val_feat = set_shared_variable(val_feat)
    test_feat = set_shared_variable(test_feat)
    train_l = set_shared_variable(train_l)
    val_l = set_shared_variable(val_l)
    test_l = set_shared_variable(test_l)

    index = T.lscalar()
    x = T.tensor3('x')
    y = T.fvector('y')  # score
    l = T.fvector('l')  # length + other nuisance variables
    feat = T.fmatrix('feat')

    emb_len_loss_const = 0.005
    emb_l1_reg = 0.01
    emb_lr = 0.001
    len_lr = 0.001

    model = LinearEvalModel(input=x, feat=feat, emb_dim=emb_dim, batch_size=batch_size, init_mean=init_mean,
                            init_range=init_range, \
                            mean_length=mean_length, feat_dim=feat_dim)

    score_cost = model.squared_error(y) + l2reg * model.l2_regularization() + l1reg * model.l1_regularization()

    get_output = theano.function(
        inputs=[],
        outputs=model.output,
        givens={
            x: test_x
            # eat: test_feat
        },
        on_unused_input='warn'
    )

    get_output_val = theano.function(
        inputs=[],
        outputs=model.output,
        givens={
            x: val_x
            # feat: val_feat
        },
        on_unused_input='warn'
    )

    get_output_train = theano.function(
        inputs=[],
        outputs=model.output,
        givens={
            x: train_x
            # feat: train_feat
        },
        on_unused_input='warn'
    )

    learning_rate = np.float32(learning_rate)
    emb_lr = np.float32(emb_lr)
    len_lr = np.float32(len_lr)
    g_M = T.grad(cost=score_cost, wrt=model.M)
    g_N = T.grad(cost=score_cost, wrt=model.N)
    #    g_q2 = T.grad(cost=len_cost, wrt=model.q2)
    #    g_b2 = T.grad(cost=len_cost, wrt=model.b2)

    updates = [(model.M, model.M - learning_rate * g_M),
               (model.N, model.N - learning_rate * g_N)]
    #                (model.q2, model.q2 - learning_rate * g_q2),
    #                (model.b2, model.b2 - learning_rate * g_b2),
    '''
    updates = { model.M: model.M - learning_rate * g_M,
                model.N: model.N - learning_rate * g_N, 
                model.F: model.F - learning_rate * g_F, 
                #(model.q, model.q - learning_rate_q * g_q), 
                #(model.b, model.b - learning_rate * g_b), 
                model.f: model.f - learning_rate * g_f }
    '''
    # net_params = lasagne.layers.get_all_params(model.net, trainable=True)
    # net_updates = lasagne.updates.adam(len_cost, net_params, learning_rate=learning_rate_len)
    # print net_updates
    # updates.update(net_updates)

    train_model = theano.function(
        inputs=[index],
        outputs=score_cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size],
            l: train_l[index * batch_size: (index + 1) * batch_size],
            feat: train_feat[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='warn'

    )

    print('Compiling took %f s' % (time.time() - compile_start))

    print('...starting training')
    epoch = 0
    first_output = get_output()
    first_cor = correlation(first_output, test_y)
    first_cor_val = correlation(get_output_val(), val_y_values)
    first_cor_train = correlation(get_output_train(), train_y_values)
    best_output = np.zeros((50,))
    best_cor = [0, 0]
    best_test_cor = [0, 0]
    loss_list = []
    len_loss_list = []
    emb_loss_list = []
    spearman_train = []
    pearson_train = []
    spearman_val = []
    pearson_val = []
    spearman_test = []
    pearson_test = []
    best_correlation = -np.inf
    start_time = time.time()
    while (epoch < num_epochs):
        epoch += 1
        if epoch % 100 == 0:
            print('Starting epoch', epoch)
        cost_list = []
        for minibatch_index in range(n_train_batches):
            minibatch_cost = train_model(minibatch_index)
            cost_list.append(minibatch_cost)
        model_out = get_output()
        model_train_out = get_output_train()
        model_val_out = get_output_val()
        loss = sum(cost_list) / float(len(cost_list))
        loss_list.append(loss)

        train_correlation = correlation(model_train_out, train_y_values)
        val_correlation = correlation(model_val_out, val_y_values)
        test_correlation = correlation(model_out, test_y)
        spearman_train.append(train_correlation[0][0])
        spearman_val.append(val_correlation[0][0])
        spearman_test.append(test_correlation[0][0])
        pearson_train.append(train_correlation[1][0])
        pearson_val.append(val_correlation[1][0])
        pearson_test.append(test_correlation[1][0])

        if val_correlation[0][0] > best_correlation:
            best_correlation = val_correlation[0][0]
            best_cor = val_correlation
            best_test_cor = test_correlation
            best_output = get_output_val()
            best_output_test = get_output()
            best_params = model.get_params()
            # with open('best_model.pkl', 'w') as f:
            #    pickle.dump(model, f)

    end_time = time.time()

    # Print out results
    folder_name = pca_name + '_bs=' + str(batch_size) + '_lr=' + str(learning_rate) + '_l1=' + \
                  str(l1reg) + '_l2=' + str(l2reg) + '_epochs=' + str(num_epochs)
    print_string = '%%%% ' + folder_name + ' %%%%'
    print_string += '\n Finished training. Took %f s' % (end_time - start_time)
    print_string += '\n Spearman correlation (test): ' + str(best_test_cor[0])
    print_string += '\n Peason correlation (test): ' + str(best_test_cor[1])
    print_string += '\n Best Spearman correlation (val): ' + str(best_cor[0])
    print_string += '\n Best Peason correlation (val): ' + str(best_cor[1])
    print_string += '\n Final Spearman correlation (train): ' + str(train_correlation[0])
    print_string += '\n Final Peason correlation (train): ' + str(train_correlation[1])
    print_string += '\n Initial Spearman correlation (train): ' + str(first_cor_train[0])
    print_string += '\n Initial Peason correlation (train): ' + str(first_cor_train[1])
    # print_string +=  '\n Final value of q : ' + str(model.q)
    # print_string +=  '\n Initial Spearman correlation (test): ' + str(first_cor[0])
    # print_string +=  '\n Initial Peason correlation (test): ' + str(first_cor[1])
    # print_string +=  '\n Initial Spearman correlation (val): ' + str(first_cor_val[0])
    # print_string +=  '\n Initial Peason correlation (val): ' + str(first_cor_val[1])
    print(print_string)
    if not os.path.exists('./results/' + exp_folder + folder_name):
        os.makedirs('./results/' + exp_folder + folder_name)
    ''' 
    # Make scatter plots
    make_plot(best_output, test_y, './results/' + exp_folder + folder_name + '/best.png')
    make_plot(first_output, test_y, './results/' + exp_folder + folder_name + '/init.png')
    make_plot(best_output_test, test_y, './results/' + exp_folder + folder_name + '/final(test).png')
    make_plot(model_train_out, train_y_values, './results/' + exp_folder + folder_name + '/final(train).png')
    test_y_gauss = []
    for y in test_y:
        test_y_gauss.append(y + np.random.normal(0, 0.3))
    make_plot(best_output, test_y_gauss, './results/' + exp_folder + folder_name + '/best_gauss.png')
    make_plot(first_output, test_y_gauss, './results/' + exp_folder + folder_name + '/init_gauss.png')
    make_plot(model_out, test_y_gauss, './results/' + exp_folder + folder_name + '/final(test)_gauss.png')
    '''
    # Make learning curves
    epoch_list = list(range(len(loss_list)))
    make_line_plot(loss_list, epoch_list, './results/' + exp_folder + folder_name + '/loss.png')
    make_line_plot(len_loss_list, epoch_list, './results/' + exp_folder + folder_name + '/len_loss.png')
    make_line_plot(emb_loss_list, epoch_list, './results/' + exp_folder + folder_name + '/emb_loss.png')
    make_line_plot(spearman_train, epoch_list, './results/' + exp_folder + folder_name + '/spear_train.png')
    make_line_plot(spearman_test, epoch_list, './results/' + exp_folder + folder_name + '/spear_test.png')
    make_line_plot(pearson_train, epoch_list, './results/' + exp_folder + folder_name + '/pear_train.png')
    make_line_plot(pearson_test, epoch_list, './results/' + exp_folder + folder_name + '/pear_test.png')

    '''
    # Save model predictions
    predictions = []
    predictions.append(("Context", "GT response", "Model response", "ADEM prediction", "Human score", "BLEU2 score", "ROUGE score"))
    for i in xrange(len(test_y)):
        predictions.append((test_contexts[i], test_gtresponses[i], test_modelresponses[i], model_out[i], test_y[i], bleu2_list[i], rouge_list[i]))
    with open('./results/' + exp_folder + folder_name + '/predictions.csv', 'w') as f1:
        writer = csv.writer(f1)
        for pred in predictions:
            writer.writerow(pred)
    '''

    # Save summary info
    with open('./results/' + exp_folder + folder_name + '/results.txt', 'w') as f1:
        f1.write(print_string)
    # Save best parameters
    pickle.dump(best_params, open('./results/' + exp_folder + folder_name + '/params.pkl', 'wb'))

    return '\n\n' + print_string, folder_name, best_params, model


def test(x_data, y_data, feat_data, best_params, model, exp_folder, folder_name, exp_name=''):
    """ 
    After training, tests model on new data
    """

    x_data = set_shared_variable(x_data)
    feat_data = set_shared_variable(feat_data)

    get_output_new = theano.function(
        inputs=[],
        outputs=model.output,
        givens={
            model.x: x_data,  # TODO: figure out why this doesn't work with just x:
            model.feat: feat_data
        }
    )

    model.set_params(best_params)
    scores = get_output_new()
    cor = correlation(scores, y_data)

    print_string = ' - ' * 20
    print_string += '\n Correlation for ' + exp_name + ': ' + str(cor[0])
    print_string += '\n Correlation for ' + exp_name + ': ' + str(cor[1])
    print(print_string)

    make_plot(scores, y_data, './results/' + exp_folder + folder_name + '/correlation_' + exp_name + '.png')
    with open('./results/' + exp_folder + folder_name + '/results.txt', 'wb') as f1:
        f1.write(print_string)

    return print_string


if __name__ == '__main__':
    val_pct = 0.15
    test_pct = 0.15
    use_aux_features = False
    use_precomputed_embeddings = True
    eval_overlap_metrics = False
    use_precomputed_embeddings_liu = True
    test_liu_data = True
    use_precomputed_embeddings_ubuntu = True
    test_ubuntu_data = False
    filtname = None  # which model to leave out for training
    training_pct = 1  # what percentage of (training) data to train on
    shuffle = True
    print('Loading data...')

    ubuntu_file = '../ubuntu_human_data.csv'
    twitter_file_liu = '../twitter_human_data.csv'
    clean_data_file = '../clean_data.pkl'  # Dictionary with userid as key, list of dicts as values, where each
    # dict represents a single context (c_id is the key for looking up contexts in context.pkl)
    twitter_file = '../contexts.pkl'  # List of the form [context_id, context, resp1, resp2, resp3, resp4]
    twitter_gt_file = '../true.txt'  # File with ground-truth responses. Line no. corresponds to context_id
    # New data from second round with Mike
    twitter_file2 = '../contexts_new.pkl'
    twitter_gt_file2 = '../true_new.txt'
    clean_data_file2 = '../clean_data_new.pkl'

    long_shuffle_file = './long_shuffle.pkl'
    short_shuffle_file = './short_shuffle.pkl'

    if len(sys.argv) > 2:
        if sys.argv[2] != None:
            embedding_type = sys.argv[2]
    else:
        embedding_type = 'CONTEXT'  # Can be "CONTEXT" or "DECODER"
    print('Embedding type is ' + embedding_type)

    if embedding_type == 'CONTEXT':
        context_embedding_file = './embeddings/context_emb_vhredcontext.pkl'
        modelresponses_embedding_file = './embeddings/modelresponses_emb_vhredcontext.pkl'
        gtresponses_embedding_file = './embeddings/gtresponses_emb_vhredcontext.pkl'
        context_embedding_file_liu = './embeddings/context_emb_vhredcontext_liu.pkl'
        modelresponses_embedding_file_liu = './embeddings/modelresponses_emb_vhredcontext_liu.pkl'
        gtresponses_embedding_file_liu = './embeddings/gtresponses_emb_vhredcontext_liu.pkl'
        context_embedding_file_ubuntu = './embeddings/context_emb_vhredcontext_ubuntu.pkl'
        modelresponses_embedding_file_ubuntu = './embeddings/modelresponses_emb_vhredcontext_ubuntu.pkl'
        gtresponses_embedding_file_ubuntu = './embeddings/gtresponses_emb_vhredcontext_ubuntu.pkl'
    elif embedding_type == 'DECODER':
        context_embedding_file = './embeddings/context_emb_vhreddecoder.pkl'
        modelresponses_embedding_file = './embeddings/modelresponses_emb_vhreddecoder.pkl'
        gtresponses_embedding_file = './embeddings/gtresponses_emb_vhreddecoder.pkl'

    twitter_bpe_dictionary = '../TwitterData/BPE/Twitter_Codes_5000.txt'
    twitter_bpe_separator = '@@'
    twitter_model_dictionary = '../TwitterData/BPE/Dataset.dict.pkl'
    twitter_model_prefix = '/home/ml/rlowe1/TwitterData/hred_twitter_models/1470516214.08_TwitterModel__405001'
    # twitter_model_prefix = '../TwitterModel/1470516214.08_TwitterModel__405001'
    # previously: '../TwitterModel/1470516214.08_TwitterModel__405001'
    # changed due to disk space limitations on Ryan's machine

    # Load model_map. Dictionary of the form {context_id: ['hred','tfidf', 'de', 'human']} (model order)
    model_map_file = '../models.pkl'
    model_map_file2 = '../models_new.pkl'
    with open(model_map_file, 'rb') as f1:
        model_map = pickle.load(f1)
    with open(model_map_file2, 'rb') as f1:
        model_map2 = pickle.load(f1)
    # Any context_id>988 is invalid for model_map2, according to mike
    model_map2 = fixmodelmap2(model_map2)
    model_map = dict(model_map, **model_map2)  # concatenate the dictionaries together

    # Load Twitter evaluation data from .pkl files
    twitter_contexts, twitter_gtresponses, twitter_modelresponses, twitter_human_scores, valid_contextids = get_twitter_data(
        clean_data_file, \
        twitter_file, twitter_gt_file)
    twitter_contexts2, twitter_gtresponses2, twitter_modelresponses2, twitter_human_scores2, valid_contextids2 = get_twitter_data(
        clean_data_file2, \
        twitter_file2, twitter_gt_file2)
    valid_contextids = combine_contextids(valid_contextids, valid_contextids2)
    model_map = filter_modelmap(model_map, valid_contextids)

    # Adding first round of data to new round of data
    twitter_contexts = clean_list(twitter_contexts + twitter_contexts2)
    twitter_gtresponses = clean_list(twitter_gtresponses + twitter_gtresponses2)
    twitter_modelresponses = clean_list(twitter_modelresponses + twitter_modelresponses2)
    twitter_human_scores += twitter_human_scores2


    # This is to make sure the vanilla importing of the data is not biased, and it gets the 
    # same results when shuffled
    def shuffle_dataset(contexts, gts, models, scores, is_list=True):
        if not os.path.exists(short_shuffle_file):
            shuffled_order = list(range(len(contexts)))
            random.shuffle(shuffled_order)
            long_shuffled_order = [4 * i for i in shuffled_order]
            lso = []
            for i in range(len(long_shuffled_order)):
                num = long_shuffled_order[i]
                lso.append(num)
                lso.append(num + 1)
                lso.append(num + 2)
                lso.append(num + 3)
            with open(short_shuffle_file, 'wb') as f1:
                pickle.dump(shuffled_order, f1)
            with open(long_shuffle_file, 'wb') as f1:
                pickle.dump(lso, f1)

        else:
            with open(short_shuffle_file, 'rb') as f1:
                shuffled_order = pickle.load(f1)
            with open(long_shuffle_file, 'rb') as f1:
                lso = pickle.load(f1)

        contexts, gts, models, scores = np.array(contexts), np.array(gts), np.array(models), np.array(scores)
        contexts = contexts[shuffled_order]
        gts = gts[shuffled_order]
        models = models[lso]
        scores = scores[lso]
        if is_list:
            return list(contexts), list(gts), list(models), list(scores)
        else:
            return contexts, gts, models


    def short_shuffle(ar_in):
        with open(short_shuffle_file, 'rb') as f1:
            so = pickle.load(f1)
        return ar_in[so]


    def long_shuffle(ar_in):
        with open(long_shuffle_file, 'rb') as f1:
            lso = pickle.load(f1)
        return ar_in[lso]


    if shuffle:
        twitter_contexts, twitter_gtresponses, twitter_modelresponses, twitter_human_scores = shuffle_dataset(
            twitter_contexts, \
            twitter_gtresponses, twitter_modelresponses, twitter_human_scores)

    '''
    print ' * '*30
    bad_len = []
    good_len = []
    for i in xrange(len(twitter_human_scores)):
        l = len(twitter_modelresponses[i])
        if twitter_human_scores[i] <= 2:
            bad_len.append(l)
        elif twitter_human_scores[i] >= 4:
            good_len.append(l)

    print ' * '*30
    '''

    with open('contexts_txt.txt', 'w') as f1:
        for c in twitter_contexts:
            f1.write(c + '\n')
    with open('modelresponses_txt.txt', 'w') as f1:
        for r in twitter_modelresponses:
            f1.write(r[8:] + '\n')
    with open('gtresponses_txt.txt', 'w') as f1:
        for r in twitter_gtresponses:
            f1.write(r)

    # Load in Twitter dictionaries
    twitter_bpe = BPE(open(twitter_bpe_dictionary, 'r').readlines(), twitter_bpe_separator)
    twitter_dict = pickle.load(open(twitter_model_dictionary, 'rb'))
    twitter_str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in twitter_dict])
    twitter_idx_to_str = dict([(tok_id, tok) for tok, tok_id, _, _ in twitter_dict])

    # NOTE: there is still a problem with loading the Liu et al. data
    # It doesn't get any correlation for any metrics.

    # Get Liu et al. data, for Ubuntu
    ubuntu_data = load_data(ubuntu_file)
    ubuntu_contexts = get_context(ubuntu_data)
    ubuntu_gtresponses = get_gtresponse(ubuntu_data)
    ubuntu_modelresponses = get_modelresponse(ubuntu_data)
    ubuntu_human_scores = get_score(ubuntu_data)

    # Get Liu et al. data, for Twitter (with old formatting)
    twitter_data_liu = load_data(twitter_file_liu)
    twitter_contexts_liu = get_context(twitter_data_liu)
    twitter_gtresponses_liu = get_gtresponse(twitter_data_liu)
    twitter_modelresponses_liu = get_modelresponse(twitter_data_liu)
    twitter_human_scores_liu = get_score(twitter_data_liu)

    # print twitter_contexts_liu[0]
    # tw_ids = strs_to_idxs(twitter_contexts_liu, twitter_bpe, twitter_str_to_idx)
    # print idxs_to_strs(tw_ids, twitter_bpe, twitter_idx_to_str)[0]

    if eval_overlap_metrics:
        # print 'For our Twitter...'
        bleu2_list, rouge_list = show_overlap_scores(twitter_gtresponses, twitter_modelresponses, twitter_human_scores,
                                                     test_pct, twitter_contexts)
        print('For Liu Twitter...')
        # show_overlap_scores(ubuntu_gtresponses, ubuntu_modelresponses, ubuntu_human_scores, test_pct)
        # show_overlap_scores(twitter_gtresponses_liu, twitter_modelresponses_liu, twitter_human_scores_liu, test_pct, twitter_contexts_liu, liu=True)
    else:
        bleu2_list, rouge_list = [], []


    def preprocess_data(twitter_contexts, twitter_gtresponses, twitter_modelresponses, context_embedding_file, \
                        gtresponses_embedding_file, modelresponses_embedding_file, use_precomputed_embeddings,
                        liu=False):
        # Encode text into BPE format
        twitter_context_ids = strs_to_idxs(twitter_contexts, twitter_bpe, twitter_str_to_idx)
        twitter_gtresponse_ids = strs_to_idxs(twitter_gtresponses, twitter_bpe, twitter_str_to_idx)
        twitter_modelresponse_ids = strs_to_idxs(twitter_modelresponses, twitter_bpe, twitter_str_to_idx)

        # Compute VHRED embeddings
        if use_precomputed_embeddings:
            print('Loading precomputed embeddings...')
            with open(context_embedding_file, 'rb') as f1:
                twitter_context_embeddings = pickle.load(f1)
            with open(gtresponses_embedding_file, 'rb') as f1:
                twitter_gtresponse_embeddings = pickle.load(f1)
            with open(modelresponses_embedding_file, 'rb') as f1:
                twitter_modelresponse_embeddings = pickle.load(f1)

        elif 'gpu' in theano.config.device.lower():
            print('Loading model...')
            state = prototype_state()
            state_path = twitter_model_prefix + "_state.pkl"
            model_path = twitter_model_prefix + "_model.npz"

            with open(state_path, 'rb') as src:
                state.update(pickle.load(src))

            state['bs'] = 20
            state['dictionary'] = twitter_model_dictionary

            model = DialogEncoderDecoder(state)

            print('Computing context embeddings...')
            twitter_context_embeddings = compute_model_embeddings(twitter_context_ids, model, embedding_type)
            with open(context_embedding_file, 'wb') as f1:
                pickle.dump(twitter_context_embeddings, f1)
            print('Computing ground truth response embeddings...')
            twitter_gtresponse_embeddings = compute_model_embeddings(twitter_gtresponse_ids, model, embedding_type)
            with open(gtresponses_embedding_file, 'wb') as f1:
                pickle.dump(twitter_gtresponse_embeddings, f1)
            print('Computing model response embeddings...')
            twitter_modelresponse_embeddings = compute_model_embeddings(twitter_modelresponse_ids, model,
                                                                        embedding_type)
            with open(modelresponses_embedding_file, 'wb') as f1:
                pickle.dump(twitter_modelresponse_embeddings, f1)

        else:
            # Set embeddings to 0 for now. alternatively, we can load them from disc...
            # embeddings = pickle.load(open(embedding_file, 'rb'))
            print('ERROR: No GPU specified!')
            print(' To save testing time, model will be trained with zero context / response embeddings...')
            twitter_context_embeddings = np.zeros((len(twitter_context_embeddings), 3, emb_dim))
            twitter_gtresponses_embedding = np.zeros((len(twitter_context_embeddings), 3, emb_dim))
            twitter_modelresponse_embeddings = np.zeros((len(twitter_context_embeddings), 3, emb_dim))

        if not liu:
            # Copy the contexts and gt responses 4 times (to align with the model responses)
            if shuffle:
                twitter_context_embeddings = list(short_shuffle(np.array(twitter_context_embeddings)))
                twitter_gtresponse_embeddings = list(short_shuffle(np.array(twitter_gtresponse_embeddings)))
                twitter_modelresponse_embeddings = list(long_shuffle(np.array(twitter_modelresponse_embeddings)))
            temp_c_emb = []
            temp_gt_emb = []
            temp_gt = []
            for i in range(len(twitter_context_embeddings)):
                temp_c_emb.append([twitter_context_embeddings[i]] * 4)
                temp_gt_emb.append([twitter_gtresponse_embeddings[i]] * 4)
                temp_gt.append([twitter_gtresponses[i]] * 4)
            twitter_context_embeddings = flatten(temp_c_emb)
            twitter_gtresponse_embeddings = flatten(temp_gt_emb)
            twitter_gtresponses = flatten(temp_gt)

        assert len(twitter_context_embeddings) == len(twitter_gtresponse_embeddings)
        assert len(twitter_context_embeddings) == len(twitter_modelresponse_embeddings)

        emb_dim = twitter_context_embeddings[0].shape[0]

        twitter_dialogue_embeddings = np.zeros((len(twitter_context_embeddings), 3, emb_dim))
        for i in range(len(twitter_context_embeddings)):
            twitter_dialogue_embeddings[i, 0, :] = twitter_context_embeddings[i]
            twitter_dialogue_embeddings[i, 1, :] = twitter_gtresponse_embeddings[i]
            twitter_dialogue_embeddings[i, 2, :] = twitter_modelresponse_embeddings[i]

        print('Computing auxiliary features...')
        if use_aux_features:
            aux_features = get_auxiliary_features(twitter_contexts, twitter_gtresponses, twitter_modelresponses,
                                                  len(twitter_modelresponses))
            # if not liu and shuffle:
            #    aux_features = long_shuffle(aux_features)
        else:
            aux_features = np.zeros((len(twitter_modelresponses), 5))

        nuisance_features = get_len_features(twitter_modelresponses, len(twitter_modelresponses))
        # if not liu and shuffle:
        #    nuisance_features = long_shuffle(nuisance_features)

        return twitter_dialogue_embeddings, nuisance_features, aux_features


    twitter_dialogue_embeddings, nuisance_features, aux_features = preprocess_data(twitter_contexts,
                                                                                   twitter_gtresponses,
                                                                                   twitter_modelresponses,
                                                                                   context_embedding_file, \
                                                                                   gtresponses_embedding_file,
                                                                                   modelresponses_embedding_file,
                                                                                   use_precomputed_embeddings)

    if test_liu_data:
        print('Preprocessing Liu et al. Twitter data...')
        twitter_dialogue_embeddings_liu, _, liu_feat = preprocess_data(twitter_contexts_liu, twitter_gtresponses_liu,
                                                                       twitter_modelresponses_liu, \
                                                                       context_embedding_file_liu,
                                                                       gtresponses_embedding_file_liu,
                                                                       modelresponses_embedding_file_liu,
                                                                       use_precomputed_embeddings_liu, liu=True)

    if test_ubuntu_data:
        print('Preprocessing Ubuntu data...')
        ubuntu_embeddings, _, ubuntu_feat = preprocess_data(ubuntu_contexts, ubuntu_gtresponses, ubuntu_modelresponses, \
                                                            context_embedding_file_ubuntu,
                                                            gtresponses_embedding_file_ubuntu,
                                                            modelresponses_embedding_file_ubuntu,
                                                            use_precomputed_embeddings_ubuntu, liu=True)

    mean_nuis = np.mean(nuisance_features)
    print("  Average length of a response is: %f characters" % (mean_nuis))
    print("  Average loss/ example on whole dataset: %f" % (np.mean((nuisance_features - mean_nuis) ** 2)))

    emb_dim = twitter_dialogue_embeddings[0, 0].shape[0]
    if len(sys.argv) > 0 and sys.argv[1] != None:
        exp_name = sys.argv[1]
    else:
        exp_name = 'rand_exp_' + str(randint(0, 99))
    exp_folder = exp_name + '/'

    train_index = int((1 - (val_pct + test_pct)) * twitter_dialogue_embeddings.shape[0])
    val_index = int((1 - test_pct) * twitter_dialogue_embeddings.shape[0])

    if filtname != None:
        train_filt = construct_filter(model_map, filtname, 'train')[:train_index]
        val_filt = construct_filter(model_map, filtname, 'val')
        test_filt = val_filt[val_index:]
        val_filt = val_filt[train_index:val_index]

    # Main loop through hyperparameter search
    separate_pca = False
    total_summary = ''
    pca_list = [5, 7, 10, 15, 20]  # , 35, 50, 100]
    l1reg_list = [0.005, 0.01, 0.02, 0.03, 0.05]  # 1e-3, 1e-2, 0.1]
    pca_list = [20, 35, 50, 75, 100, 200]
    l1reg_list = [0.0005, 0.001, 0.002, 0.005]
    pca_list = [7]  # , 20, 50, 100]
    l1reg_list = [0.02]
    l2reg_list = [0]
    lr_list = [0.01]
    last_pca = 0
    for pca_components in pca_list:
        for lr in lr_list:
            for l2reg in l2reg_list:
                for l1reg in l1reg_list:
                    print('\n%%%%%%%%%%%%%  Running experiment with PCA=' + str(pca_components) + ', l2reg=' \
                          + str(l2reg) + ', l1reg=' + str(l1reg) + ', aux_feat=' + str(use_aux_features) + \
                          ' %%%%%%%%%%%%%%')
                    if separate_pca == False:
                        print('Also, combined PCA with ' + embedding_type + ' embeddings')

                    # Separate into train and test (for the embedding data, this is done inside
                    # the PCA function
                    train_y = np.array(twitter_human_scores[:train_index])
                    val_y = np.array(twitter_human_scores[train_index:val_index])
                    test_y = np.array(twitter_human_scores[val_index:])

                    # Reduce the dimensionality of the embeddings with PCA
                    if pca_components == last_pca:
                        print('Using embeddings from last round...')
                    else:
                        print('Computing PCA...')
                        if pca_components < emb_dim:
                            if separate_pca:
                                twitter_dialogue_embeddings2 = compute_separate_pca(pca_components,
                                                                                    twitter_dialogue_embeddings)
                                train_x = twitter_dialogue_embeddings2[:train_index]
                                val_x = twitter_dialogue_embeddings2[train_index:val_index]
                                test_x = twitter_dialogue_embeddings2[val_index:]
                                pca_prefix = 'sep'
                            else:
                                train_x, val_x, test_x, pca = compute_pca(pca_components, twitter_dialogue_embeddings,
                                                                          train_index, val_index)
                                pca_prefix = ''
                            if test_liu_data:
                                liu_x = compute_liu_pca(pca_components, twitter_dialogue_embeddings_liu, pca)
                            if test_ubuntu_data:
                                ubuntu_x = compute_liu_pca(pca_components, ubuntu_embeddings, pca)
                        else:
                            twitter_dialogue_embeddings2 = twitter_dialogue_embeddings
                            pca_prefix = ''
                    init_mean, init_range = compute_init_values(train_x)

                    train_feat = aux_features[:train_index]
                    val_feat = aux_features[train_index:val_index]
                    test_feat = aux_features[val_index:]
                    train_nuis = nuisance_features[:train_index]
                    val_nuis = nuisance_features[train_index:val_index]
                    test_nuis = nuisance_features[val_index:]
                    if filtname != None:
                        train_x, train_y, train_feat = apply_train_filter(train_x, train_y, train_feat, train_filt)
                        # val_x, val_y, val_feat, test_x, test_y, test_feat = apply_test_filter(val_x, val_y, val_feat, test_x, test_y, test_feat, val_filt, test_filt)

                    if training_pct != 1:
                        train_x = train_x[:int(training_pct * train_index)]
                        train_y = train_y[:int(training_pct * train_index)]
                        train_feat = train_feat[:int(training_pct * train_index)]
                    print('Training model...')
                    summary, folder_name, best_params, model = train(train_x, val_x, test_x, train_y, val_y, test_y,
                                                                     train_nuis, val_nuis, test_nuis, \
                                                                     init_mean, init_range, learning_rate=lr,
                                                                     l2reg=l2reg, l1reg=l1reg, \
                                                                     train_feat=train_feat, val_feat=val_feat,
                                                                     test_feat=test_feat,
                                                                     pca_name=pca_prefix + 'pca' + str(pca_components),
                                                                     exp_folder=exp_folder, \
                                                                     test_contexts=extend_by_four(twitter_contexts)[
                                                                                   val_index:],
                                                                     test_modelresponses=twitter_modelresponses[
                                                                                         val_index:], \
                                                                     test_gtresponses=extend_by_four(
                                                                         twitter_gtresponses)[val_index:],
                                                                     bleu2_list=bleu2_list, rouge_list=rouge_list, \
                                                                     mean_length=mean_nuis)
                    total_summary += summary
                    if test_liu_data:
                        liu_summary = test(liu_x, twitter_human_scores_liu, liu_feat, best_params, model, exp_folder,
                                           folder_name, exp_name='liu')
                        total_summary += liu_summary
                    if test_ubuntu_data:
                        ubuntu_summary = test(ubuntu_x, ubuntu_human_scores, ubuntu_feat, best_params, model,
                                              exp_folder, folder_name, exp_name='ubuntu')
                        total_summary += ubuntu_summary
                    last_pca = pca_components

    with open('./results/summary_of_experiment_' + exp_name + '.txt', 'w') as f1:
        f1.write(total_summary)

    # Start training with:
    #   THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python eval_model_hred.py
    # or
    #   THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python eval_model_hred.py
