import os
import pathlib
import random
import json
import numpy as np
from scipy import linalg
import sklearn.cluster

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel
)

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

# ----- tools about configuring model and loading data ----- #

def read_data(file_path):
    querys, answers = [], []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            qa = line.split('\t')
            query, answer = qa[:2] if len(qa) > 1 else (qa[0], '')
            querys.append(query.strip())
            answers.append(answer.strip())
    return querys, answers

def get_model_configs(pretrained_model_path, is_chinese=False):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    model = AutoModel.from_pretrained(pretrained_model_path, return_dict=True)
    return tokenizer, model

def get_embeddings(querys, answers, tokenizer, model, batch_size, use_cuda=True):
    feats = []
    model.eval()
    if use_cuda:
        model.to('cuda')

    with torch.no_grad():
        num_batches = len(querys) // batch_size
        if len(querys) % batch_size > 0:
            num_batches += 1
        for i in tqdm(range(num_batches)):
            query = querys[i*batch_size : (i+1)*batch_size]
            answer = answers[i*batch_size : (i+1)*batch_size]
            if len(query)== 0 or len(answer) == 0:
                continue 
            inputs = tokenizer(query, answer, return_tensors='pt', padding=True, truncation=True)
            if use_cuda:
                inputs.to('cuda')
            outputs = model(**inputs)
            feats.append(outputs.last_hidden_state[:, 0].cpu().data)
    feats = torch.cat(feats).numpy()
    
    return feats


# ----- details on calculating FBD ----- #

def calculate_feature_statistics(feats):
    """Calculation of the statistics used by the FID.
    Params:
    -- feats       : tensor of features with the shape [N, D]
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


# ----- details on calculating PRD ----- # 

def compute_prd(eval_dist, ref_dist, num_angles=1001, epsilon=1e-10):
    """Computes the PRD curve for discrete distributions.
    This function computes the PRD curve for the discrete distribution eval_dist
    with respect to the reference distribution ref_dist. This implements the
    algorithm in [arxiv.org/abs/1806.2281349]. The PRD will be computed for an
    equiangular grid of num_angles values between [0, pi/2].
    Args:
      eval_dist: 1D NumPy array or list of floats with the probabilities of the
                 different states under the distribution to be evaluated.
      ref_dist: 1D NumPy array or list of floats with the probabilities of the
                different states under the reference distribution.
      num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                  The default value is 1001.
      epsilon: Angle for PRD computation in the edge cases 0 and pi/2. The PRD
               will be computes for epsilon and pi/2-epsilon, respectively.
               The default value is 1e-10.
    Returns:
      precision: NumPy array of shape [num_angles] with the precision for the
                 different ratios.
      recall: NumPy array of shape [num_angles] with the recall for the different
              ratios.
    Raises:
      ValueError: If not 0 < epsilon <= 0.1.
      ValueError: If num_angles < 3.
    """

    if not (epsilon > 0 and epsilon < 0.1):
        raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
    if not (num_angles >= 3 and num_angles <= 1e6):
        raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

    # Compute slopes for linearly spaced angles between [0, pi/2]
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=num_angles)
    slopes = np.tan(angles)

    # Broadcast slopes so that second dimension will be states of the distribution
    slopes_2d = np.expand_dims(slopes, 1)

    # Broadcast distributions so that first dimension represents the angles
    ref_dist_2d = np.expand_dims(ref_dist, 0)
    eval_dist_2d = np.expand_dims(eval_dist, 0)

    # Compute precision and recall for all angles in one step via broadcasting
    precision = np.minimum(ref_dist_2d * slopes_2d, eval_dist_2d).sum(axis=1)
    recall = precision / slopes

    # handle numerical instabilities leaing to precision/recall just above 1
    max_val = max(np.max(precision), np.max(recall))
    if max_val > 1.001:
        raise ValueError('Detected value > 1.001, this should not happen.')
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)

    return precision, recall


def _prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10):
    """Computes F_beta scores for the given precision/recall values.
    The F_beta scores for all precision/recall pairs will be computed and
    returned.
    For precision p and recall r, the F_beta score is defined as:
    F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)
    Args:
        precision: 1D NumPy array of precision values in [0, 1].
        recall: 1D NumPy array of precision values in [0, 1].
        beta: Beta parameter. Must be positive. The default value is 1.
        epsilon: Small constant to avoid numerical instability caused by division
                         by 0 when precision and recall are close to zero.
    Returns:
        NumPy array of same shape as precision and recall with the F_beta scores for
        each pair of precision/recall.
    Raises:
        ValueError: If any value in precision or recall is outside of [0, 1].
        ValueError: If beta is not positive.
    """

    if not ((precision >= 0).all() and (precision <= 1).all()):
        raise ValueError('All values in precision must be in [0, 1].')
    if not ((recall >= 0).all() and (recall <= 1).all()):
        raise ValueError('All values in recall must be in [0, 1].')
    if beta <= 0:
        raise ValueError('Given parameter beta %s must be positive.' % str(beta))

    return (1 + beta**2) * (precision * recall) / (
            (beta**2 * precision) + recall + epsilon)


def prd_to_max_f_beta_pair(precision, recall, beta=8):
    """Computes max. F_beta and max. F_{1/beta} for precision/recall pairs.
    Computes the maximum F_beta and maximum F_{1/beta} score over all pairs of
    precision/recall values. This is useful to compress a PRD plot into a single
    pair of values which correlate with precision and recall.
    For precision p and recall r, the F_beta score is defined as:
    F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)
    Args:
        precision: 1D NumPy array or list of precision values in [0, 1].
        recall: 1D NumPy array or list of precision values in [0, 1].
        beta: Beta parameter. Must be positive. The default value is 8.
    Returns:
        f_beta: Maximum F_beta score.
        f_beta_inv: Maximum F_{1/beta} score.
    Raises:
        ValueError: If beta is not positive.
    """

    if not ((precision >= 0).all() and (precision <= 1).all()):
        raise ValueError('All values in precision must be in [0, 1].')
    if not ((recall >= 0).all() and (recall <= 1).all()):
        raise ValueError('All values in recall must be in [0, 1].')
    if beta <= 0:
        raise ValueError('Given parameter beta %s must be positive.' % str(beta))

    f_beta = np.max(_prd_to_f_beta(precision, recall, beta))
    f_beta_inv = np.max(_prd_to_f_beta(precision, recall, 1/beta))
    return f_beta, f_beta_inv


def _cluster_into_bins(eval_data, ref_data, num_clusters):
    """Clusters the union of the data points and returns the cluster distribution.
    Clusters the union of eval_data and ref_data into num_clusters using minibatch
    k-means. Then, for each cluster, it computes the number of points from
    eval_data and ref_data.
    Args:
        eval_data: NumPy array of data points from the distribution to be evaluated.
        ref_data: NumPy array of data points from the reference distribution.
        num_clusters: Number of cluster centers to fit.
    Returns:
        Two NumPy arrays, each of size num_clusters, where i-th entry represents the
        number of points assigned to the i-th cluster.
    """

    cluster_data = np.vstack([eval_data, ref_data])
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit(cluster_data).labels_

    eval_labels = labels[:len(eval_data)]
    ref_labels = labels[len(eval_data):]

    eval_bins = np.histogram(eval_labels, bins=num_clusters,
                             range=[0, num_clusters], density=True)[0]
    ref_bins = np.histogram(ref_labels, bins=num_clusters,
                            range=[0, num_clusters], density=True)[0]
    return eval_bins, ref_bins


def compute_prd_from_embedding(
    eval_data, 
    ref_data, 
    num_clusters=20,
    num_angles=1001, 
    num_runs=10,
    enforce_balance=True
):
    """Computes PRD data from sample embeddings.
    The points from both distributions are mixed and then clustered. This leads
    to a pair of histograms of discrete distributions over the cluster centers
    on which the PRD algorithm is executed.
    The number of points in eval_data and ref_data must be equal since
    unbalanced distributions bias the clustering towards the larger dataset. The
    check can be disabled by setting the enforce_balance flag to False (not
    recommended).
    Args:
        eval_data: NumPy array of data points from the distribution to be evaluated.
        ref_data: NumPy array of data points from the reference distribution.
        num_clusters: Number of cluster centers to fit. The default value is 20.
        num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                                The default value is 1001.
        num_runs: Number of independent runs over which to average the PRD data.
        enforce_balance: If enabled, throws exception if eval_data and ref_data do
                                         not have the same length. The default value is True.
    Returns:
        precision: NumPy array of shape [num_angles] with the precision for the
                             different ratios.
        recall: NumPy array of shape [num_angles] with the recall for the different
                        ratios.
    Raises:
        ValueError: If len(eval_data) != len(ref_data) and enforce_balance is set to
                                True.
    """

    if enforce_balance and len(eval_data) != len(ref_data):
        raise ValueError(
                'The number of points in eval_data %d is not equal to the number of '
                'points in ref_data %d. To disable this exception, set enforce_balance '
                'to False (not recommended).' % (len(eval_data), len(ref_data)))

    eval_data = np.array(eval_data, dtype=np.float64)
    ref_data = np.array(ref_data, dtype=np.float64)
    precisions = []
    recalls = []
    for _ in range(num_runs):
        eval_dist, ref_dist = _cluster_into_bins(eval_data, ref_data, num_clusters)
        precision, recall = compute_prd(eval_dist, ref_dist, num_angles)
        precisions.append(precision)
        recalls.append(recall)
    precision = np.mean(precisions, axis=0)
    recall = np.mean(recalls, axis=0)
    return precision, recall


# ----- details on data transformation ----- #

def read_vocab(file):
    vocab = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            vocab.append(line[0])
    return vocab

def read_dict(file):
    with open(file, 'r', encoding='utf-8') as f:
        res = [line.strip() for line in f]
    return res

def read_dialogue(path):
    querys = []
    refs = []
    hyps = []
    human_scores = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            querys.append(line['src'])
            refs.append(line['refs'][0])

            for i, hyp in enumerate(line['hyps']):
                if len(hyps) < i + 1:
                    hyps.append([])
                hyps[i].append(hyp)

            for i, scores in enumerate(line['human_scores']):
                if len(human_scores) < i + 1:
                    human_scores.append([])
                for j, score in enumerate(scores):
                    if len(human_scores[i]) < j + 1:
                        human_scores[i].append([])
                    human_scores[i][j].append(score)
    
    return querys, refs, hyps, human_scores

def transform_qa_pairs(querys, answers, transform, ratio, noise_dict, repeat_dict):
    trans_answers = list(answers)
    trans_indexs = random.sample(list(range(len(querys))), int(len(querys) * ratio))

    if transform == 'noise':
        assert noise_dict != None
        vocab = read_vocab(noise_dict)
        for trans_index in trans_indexs:
            trans_answer = answers[trans_index].split()
            num_lower = len(trans_answer) // 4
            num_upper = max(len(trans_answer) // 2 + 1, len(trans_answer) // 4 + 2)
            num_list = list(range(num_lower, num_upper))
            num = random.choice(num_list)
            for _ in range(num):
                loc = random.randint(0, len(trans_answer))
                word = random.choice(vocab)
                trans_answer.insert(loc, word)
            trans_answers[trans_index] = ' '.join(trans_answer)

    elif transform == 'mismatch':
        indexs = sorted(trans_indexs)
        for index, trans_index in zip(indexs, trans_indexs):
            trans_answers[index] = answers[trans_index]

    elif transform == 'permutate':
        for trans_index in trans_indexs:
            trans_answer = answers[trans_index].split()
            random.shuffle(trans_answer)
            trans_answers[trans_index] = ' '.join(trans_answer)
    
    elif transform == 'repeat':
        assert repeat_dict != None
        repeat_dict = read_dict(repeat_dict)
        for trans_index in trans_indexs:
            trans_answers[trans_index] = random.choice(repeat_dict)

    else:
        raise RuntimeError('Unknown transformation: {}'.format(transform))

    return querys, trans_answers