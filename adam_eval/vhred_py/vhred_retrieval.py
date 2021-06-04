import csv

import matplotlib
import numpy as np

matplotlib.use('Agg')
import theano
import time
import math
import pickle

from .vhred_dialog_encdec import DialogEncoderDecoder
from .vhred_state import prototype_state
from .vhred_compute_dialogue_embeddings import compute_encodings

import os

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


# Computes PCA decomposition for the context, gt responses, and model responses separately
def compute_separate_pca(pca_components, twitter_dialogue_embeddings):
    pca = PCA(n_components=pca_components)
    tw_embeddings_pca = np.zeros((twitter_dialogue_embeddings.shape[0], 3, pca_components))
    for i in range(3):
        tw_embeddings_pca[:, i] = pca.fit_transform(twitter_dialogue_embeddings[:, i])
    return tw_embeddings_pca


# Computes PCA decomposition for the context, gt responses, and model responses together
def compute_pca(pca_components, twitter_dialogue_embeddings):
    pca = PCA(n_components=pca_components)
    num_ex = twitter_dialogue_embeddings.shape[0]
    dim = twitter_dialogue_embeddings.shape[2]
    tw_embeddings_pca = np.zeros((num_ex * 3, dim))
    for i in range(3):
        tw_embeddings_pca[num_ex * i: num_ex * (i + 1), :] = twitter_dialogue_embeddings[:, i]
    tw_embeddings_pca = pca.fit_transform(tw_embeddings_pca)
    tw_emb = np.zeros((num_ex, 3, pca_components))
    for i in range(3):
        tw_emb[:, i] = tw_embeddings_pca[num_ex * i: num_ex * (i + 1), :]
    return tw_emb


def idxs_to_strs(data, bpe, idx_to_str):
    ''' Converts from BPE form to strings '''
    out = []
    for row in data:
        out.append(' '.join([idx_to_str[idx] for idx in row if idx in idx_to_str]).replace('@@ ', ''))
    return out


def flatten_list(l1):
    return [i for sublist in l1 for i in sublist]


# Compute model embeddings for contexts or responses
# Embedding type can be 'CONTEXT' or 'DECODER'
def compute_model_embeddings(data, model, embedding_type, ftype, starting_batch=0, max_batches=np.infty):
    model_compute_encoding = model.build_encoder_function()
    model_compute_decoder_encoding = model.build_decoder_encoding()

    embeddings = []
    context_ids_batch = []
    batch_index = starting_batch
    batch_total = int(math.ceil(float(len(data)) / float(model.bs)))
    counter = 0
    fcounter = 0
    start = time.time()
    # TODO: remove temporary code
    data = data[starting_batch * model.bs:]
    fcounter = int(starting_batch / 1000.0) + 1  # int(max_batches / 1000.0 / 2.0) + 2
    print('This code will first write to ' + ftype + '_emb' + str(fcounter) + '.pkl')
    ###
    for context_ids in data:
        context_ids_batch.append(context_ids)
        counter += 1
        if len(context_ids_batch) == model.bs or counter == len(data):
            batch_index += 1

            print('     Computing embeddings for batch ' + str(batch_index) + ' / ' + str(batch_total), end=' ')
            encs = compute_encodings(context_ids_batch, model, model_compute_encoding, model_compute_decoder_encoding,
                                     embedding_type)
            for i in range(len(encs)):
                embeddings.append(encs[i])

            context_ids_batch = []
            print(time.time() - start)
            start = time.time()

            if batch_index % 1000 == 0 or counter == len(data):
                fcounter += 1
                if embedding_type == 'CONTEXT':
                    pickle.dump(embeddings, open(
                        '/home/ml/rlowe1/TwitterData/vhred_context_emb/' + ftype + '_emb' + str(fcounter) + '.pkl',
                        'w'))
                elif embedding_type == 'DECODER':
                    pickle.dump(embeddings, open(
                        '/home/ml/rlowe1/TwitterData/vhred_decoder_emb/' + ftype + '_emb' + str(fcounter) + '.pkl',
                        'w'))
                else:
                    pickle.dump(embeddings, open(
                        '/home/ml/rlowe1/TwitterData/vhred_emb_other/' + ftype + '_emb_' + str(fcounter) + '.pkl', 'w'))
                embeddings = []

                if batch_index >= max_batches:
                    return embeddings

    return embeddings


def scale_points(train_emb, test_emb, max_val):
    '''
    Scales all points in train_emb, test_emb such that
    max_i ||x_i||_2 = max_val
    max_val corresponds to U in the Auvolat et al. paper
    '''
    pass


def transform_data_points(train_emb):
    emb = []
    pass


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


def test_model(train_emb, test_emb, train_responses, test_responses, train_contexts, test_contexts, output_file):
    '''
    Tests the model by finding the closest context embedding in the training set
    for each test query (using approximate MIPS). Then, outputs the corresponding response from
    the training set.
    Approximate MIPS is done using the spherical k-means method from Auvolat et al. (2016)    
    NOTE: Right now this just does a numpy array multiplication. No approximate MIPS is used.
    '''
    # Compute the cosine similarity between each test embedding and each embedding in the test set
    # This is done by taking the product of the test and train embedding matrices, and dividing
    # by the 2-norms of the vectors
    test_ar = np.array(test_emb)
    train_ar = np.array(train_emb)
    prod_matrix = np.dot(train_ar, test_ar.T)  # has shape (train_ex, test_ex)
    prod_matrix = prod_matrix / mat_vector_2norm(test_ar)  # divide by 2-norm of vectors to produce cosine sim
    prod_matrix = (prod_matrix.T / mat_vector_2norm(train_ar)).T
    argmax_ar = np.argmax(prod_matrix, axis=0)

    model_responses = []
    closest_contexts = []
    highest_dotproduct = []

    for train_index, example in zip(list(argmax_ar), list(range(len(argmax_ar)))):
        model_responses.append(train_responses[train_index])
        closest_contexts.append(train_contexts[train_index])
        highest_dotproduct.append(prod_matrix[train_index][example])

    # Write data to output CSV
    with open(output_file, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(['Context', 'Score', 'Model Response', 'GT Response', 'Closest Context'])
        for i in range(len(model_responses)):
            writer.writerow([test_contexts[i], highest_dotproduct[i], model_responses[i], \
                             test_responses[i], closest_contexts[i]])


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.register('type','bool',str2bool)
    # parser.add_argument('--use_precomputed_embeddings', type='bool', default=True, help='Load precomputed embeddings. If False, will re-generate and save embeddings')
    # parser.add_argument('--

    twitter_bpe_dictionary = '../TwitterData/BPE/Twitter_Codes_5000.txt'
    twitter_bpe_separator = '@@'
    twitter_model_dictionary = '../TwitterData/BPE/Dataset.dict.pkl'

    twitter_model_prefix = '/home/ml/rlowe1/TwitterData/hred_twitter_models/1470516214.08_TwitterModel__405001'
    twitter_data_prefix = '/home/ml/rlowe1/TwitterData/vhred_context_emb_old/'

    max_trainemb_index = 20  # max = 759
    max_testemb_index = 20  # max = 99
    use_precomputed_embeddings = False
    embedding_type = 'CONTEXT'  # Can be 'DECODER' or 'CONTEXT'

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

    train_contexts_txt = idxs_to_strs(train_contexts, twitter_bpe, twitter_idx_to_str)
    train_responses_txt = idxs_to_strs(train_responses, twitter_bpe, twitter_idx_to_str)
    test_contexts_txt = idxs_to_strs(test_contexts, twitter_bpe, twitter_idx_to_str)
    test_responses_txt = idxs_to_strs(test_responses, twitter_bpe, twitter_idx_to_str)

    # Encode text into BPE format
    # twitter_context_ids = strs_to_idxs(twitter_contexts, twitter_bpe, twitter_str_to_idx)
    # twitter_gtresponses_ids = strs_to_idxs(twitter_gtresponses, twitter_bpe, twitter_str_to_idx)
    # twitter_modelresponses_ids = strs_to_idxs(twitter_modelresponses, twitter_bpe, twitter_str_to_idx)

    # Compute VHRED embeddings
    if use_precomputed_embeddings:
        # Load embeddings from /home/ml/rlowe1/TwitterData/vhred_context_emb_old
        print('Loading training context embeddings...')
        train_emb = []
        for i in range(1, max_trainemb_index + 1):
            if i % 20 == 0:
                path = twitter_data_prefix + 'train_context_emb' + str(i) + '.pkl'
                with open(path, 'r') as f1:
                    train_emb.append(pickle.load(f1))

        print('Loading testing context embeddings...')
        test_emb = []
        for i in range(1, max_testemb_index + 1):
            if i % 20 == 0:
                path = twitter_data_prefix + 'test_context_emb' + str(i) + '.pkl'
                with open(path, 'r') as f1:
                    test_emb.append(pickle.load(f1))
        train_context_embeddings = flatten_list(train_emb)
        test_context_embeddings = flatten_list(test_emb)


    elif 'gpu' in theano.config.device.lower():
        state = prototype_state()
        state_path = twitter_model_prefix + "_state.pkl"
        model_path = twitter_model_prefix + "_model.npz"

        with open(state_path) as src:
            state.update(pickle.load(src))

        state['bs'] = 20
        state['dictionary'] = twitter_model_dictionary

        model = DialogEncoderDecoder(state)

        calc_response_embeddings = False
        calc_context_embeddings = True
        calc_test = False
        start_batch = 24000
        max_batches = 35000

        if calc_response_embeddings:
            print('Computing training response embeddings...')
            train_response_embeddings = compute_model_embeddings(train_responses, model, embedding_type,
                                                                 'train_response', starting_batch=start_batch,
                                                                 max_batches=max_batches)
            if calc_test:
                print('Computing test response embeddings...')
                test_response_embeddings = compute_model_embeddings(test_responses, model, embedding_type,
                                                                    'test_response', max_batches=2000)

        # Computed up to batch 22420 for DECODER
        if calc_context_embeddings:
            print('Computing training context embeddings...')
            train_context_embeddings = compute_model_embeddings(train_contexts, model, embedding_type, 'train_context',
                                                                starting_batch=start_batch, max_batches=max_batches)
            if calc_test:
                print('Computing test context embeddings...')
                test_context_embeddings = compute_model_embeddings(test_contexts, model, embedding_type, 'test_context',
                                                                   max_batches=2000)

        # assert len(train_context_embeddings) == len(test_context_embeddings)

    else:
        # Set embeddings to 0 for now. alternatively, we can load them from disc...
        # embeddings = pickle.load(open(embedding_file, 'rb'))
        print('ERROR: No GPU specified!')
    '''
    pca_components = 50
    print 'Computing PCA...'
    if pca_components < emb_dim:
        if separate_pca:
            twitter_dialogue_embeddings2 = compute_separate_pca(pca_components, twitter_dialogue_embeddings)
            pca_prefix = 'sep'
        else:
            twitter_dialogue_embeddings2 = compute_pca(pca_components, twitter_dialogue_embeddings)
            pca_prefix = ''
    else:
        twitter_dialogue_embeddings2 = twitter_dialogue_embeddings
        pca_prefix = ''

    
    pickle.dump(embeddings, open('/home/ml/rlowe1/TwitterData/context_emb_pca30/'+ftype+'_context_emb_'+str(fcounter)+'.pkl', 'w'))            
    '''
    start = time.time()
    print('Testing model...')
    test_model(train_context_embeddings, test_context_embeddings, train_responses_txt, test_responses_txt,
               train_contexts_txt, test_contexts_txt, output_file)
    print('Took %f seconds' % (time.time() - start))

###############
