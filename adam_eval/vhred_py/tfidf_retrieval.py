import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.preprocessing import *

os.sys.path.insert(0, '../TwitterData/BPE/subword_nmt')

from apply_bpe import BPE


def preprocess_tweet(s):
    s = s.replace('@user', '<at>').replace('&lt;heart&gt;', '<heart>').replace('&lt;number&gt;', '<number>').replace(
        '  ', ' </s> ').replace('  ', ' ')
    # Make sure we end with </s> token
    while s[-1] == ' ':
        s = s[0:-1]
    if not s[-5:] == ' </s>':
        s = s + ' </s>'
    return s


def process_dialogues(dialogues):
    ''' Removes </d> </s> at end, splits into contexts/ responses '''
    contexts = []
    responses = []
    for d in dialogues:
        d_proc = d[:-3]
        index_list = [i for i, j in enumerate(d_proc) if j == 1]
        split = index_list[-1] + 1
        contexts.append(d_proc[:split])
        responses.append(d_proc[split:] + [1])
    return contexts, responses


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


def idxs_to_bpestrs(data, bpe, idx_to_str):
    ''' Converts from BPE form to strings '''
    out = []
    for row in data:
        out.append(' '.join([idx_to_str[idx] for idx in row if idx in idx_to_str]))
    return out


def bpestrs_to_strs(data):
    out = []
    for row in data:
        out.append(row.replace('@@ ', ''))
    return out


def flatten_list(l1):
    return [i for sublist in l1 for i in sublist]


def brute_force_search(train_emb, query_emb):
    max_index = -1
    largest_product = -1e9
    for i in range(len(train_emb)):
        prod = np.dot(train_emb[i], query_emb)
        if prod > largest_product:
            largest_product = prod
            max_index = i
    return max_index, largest_product


def mat_vector_2norm(mat):
    '''
    Takes as input a matrix, and returns a vector correponding to the 2-norm
    of each row vector.
    '''
    norm_list = []
    for i in range(mat.shape[0]):
        norm_list.append(np.sqrt(np.dot(mat[0], mat[0].T)))
    return np.array(norm_list)


def mat_vector_2norm_squared(mat):
    '''
    Takes as input a matrix, and returns a vector correponding to the 2-norm
    of each row vector.
    '''
    norm_list = []
    for i in range(mat.shape[0]):
        norm_list.append(np.dot(mat[0], mat[0].T))
    return np.array(norm_list)


def sanity_check(test_emb, train_emb, num_test):
    '''
    Sanity check on the cosine similarity calculations
    Finds the closest vector in the space by brute force
    '''
    correct_list = []
    for i in range(num_test):
        smallest_norm = np.infty
        index = 0
        for j in range(len(train_emb)):
            norm = np.linalg.norm(emb - test_emb[i])
            if norm < smallest_norm:
                smallest_norm = norm
                index = j
        correct_list.append(index)
    # Pad the list to make it the same length as test_emb
    for i in range(len(test_emb) - num_test):
        correct_list.append(-1)
    return correct_list


def tfidf_retrieval(tfidf_vec, train_contexts_txt, train_responses_txt, output_file):
    print(type(tfidf_vec))
    tfidf_vec = tfidf_vec.toarray()
    print(tfidf_vec.shape)
    prod_mat = np.dot(tfidf_vec, tfidf_vec.T)
    print(prod_mat.shape)
    prod_mat = prod_mat / mat_vector_2norm_squared(tfidf_vec)
    print(prod_mat.shape)

    response_list = []
    for i in range(len(prod_mat)):
        row = prod_mat[i]
        # No idea what's going on here. See the following page:
        # stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        ind = np.argpartition(row, -2)[-2:]
        ind = ind[np.argsort(row[ind])][0]
        response_list.append(train_responses_txt[ind])
        print(train_contexts_txt[i])
        print(response_list[i])

    with open(output_file, 'w') as f1:
        for response in response_list:
            f1.write(response)


if __name__ == '__main__':
    twitter_bpe_dictionary = '../TwitterData/BPE/Twitter_Codes_5000.txt'
    twitter_bpe_separator = '@@'
    twitter_model_dictionary = '../TwitterData/BPE/Dataset.dict.pkl'

    # Load in Twitter dictionaries
    twitter_bpe = BPE(open(twitter_bpe_dictionary, 'r').readlines(), twitter_bpe_separator)
    twitter_dict = pickle.load(open(twitter_model_dictionary, 'r'))
    twitter_str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in twitter_dict])
    twitter_idx_to_str = dict([(tok_id, tok) for tok, tok_id, _, _ in twitter_dict])

    # Get data, for Twitter
    train_file = '/home/ml/rlowe1/TwitterData/TwitterDataBPE/Train.dialogues.pkl'
    test_file = '/home/ml/rlowe1/TwitterData/TwitterDataBPE/Test.dialogues.pkl'
    output_file = './output.csv'

    with open(train_file) as f1:
        train_data = pickle.load(f1)
    with open(test_file) as f1:
        test_data = pickle.load(f1)

    train_contexts, train_responses = process_dialogues(train_data)
    test_contexts, test_responses = process_dialogues(test_data)

    train_contexts_txt = idxs_to_bpestrs(train_contexts, twitter_bpe, twitter_idx_to_str)
    train_responses_txt = idxs_to_bpestrs(train_responses, twitter_bpe, twitter_idx_to_str)
    # test_contexts_txt = idxs_to_strs(test_contexts, twitter_bpe, twitter_idx_to_str)
    # test_responses_txt = idxs_to_strs(test_responses, twitter_bpe, twitter_idx_to_str)

    print('Fitting vectorizer...')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_contexts_txt + train_responses_txt)
    c_vec = vectorizer.fit(train_contexts_txt)
    r_vec = vectorizer.fit(train_responses_txt)

    print('Retrieving responses...')
    tfidf_retrieval(r_vec, train_contexts_txt, train_responses_txt, './rtfidf_responses.txt')
    tfidf_retrieval(c_vec, train_contexts_txt, train_responses_txt, './ctfidf_responses.txt')
