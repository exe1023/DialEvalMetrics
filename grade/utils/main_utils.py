import os
import json
import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import torch
import scipy.sparse as sp

from .normalization import fetch_normalization

def maybe_create_file(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int):
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and linear decay.
    """
    step = min(step, total_steps)

    multiplier = (1 - (step - warmup_steps) / (total_steps - warmup_steps))

    if warmup_steps > 0 and step < warmup_steps:
        warmup_percent_done = step / warmup_steps
        multiplier = warmup_percent_done

    return multiplier


def add_loss_accu_msg(args, logging, avg_rec, output_tuple, batch_size):

    losses, ranking_loss, ranking_accu, pred_preference_label, score_of_pair_1, score_of_pair_2 = output_tuple
    avg_rec.add([losses.mean(), ranking_loss.mean(), ranking_accu.mean(), \
        score_of_pair_1.mean(), score_of_pair_2.mean()], batch_size)

    return avg_rec, losses, (score_of_pair_1, score_of_pair_2)


def print_loss_accu_predlabel(args, logging, avg_rec, scores_tuple, mode, \
    summary_writer=None, iteration=-1, epoch=-1, step=-1, cur_lr=-1., batch_time=-1., left_time=-1., nsamples=-1):
    
    logging.info("mode:%s; epoch: %d; cur_lr: %f; step: %d; losses: %f; ranking_loss: %f; ranking_accu: %f; time: %f min; left_time: %f h; nsamples: %d", \
    mode, epoch, cur_lr, step, avg_rec.avg(0), avg_rec.avg(1), avg_rec.avg(2), batch_time, left_time, nsamples)

    if mode is not 'test':
        summary_writer.add_scalar('{}_losses'.format(mode), avg_rec.avg(0), iteration) # save every iteration
        summary_writer.add_scalar('{}_ranking_loss'.format(mode), avg_rec.avg(1), iteration)
        summary_writer.add_scalar('{}_ranking_accu'.format(mode), avg_rec.avg(2), iteration)
        summary_writer.add_scalar('{}_lr'.format(mode), cur_lr, iteration) # save every iteration

    score_of_pair_1, score_of_pair_2 = scores_tuple
    logging.info("score_of_pair_1: {}".format([np.round(item[0],4) for item in score_of_pair_1.tolist()]))
    logging.info("score_of_pair_2: {}".format([np.round(item[0],4) for item in score_of_pair_2.tolist()]))


def print_evaluation_results(metrics):
    print("=================== Printing GRADE metric Result ===================")
    for name, value in metrics.items():
        print ('[{} mean score:] {:.4f}'.format(name, value))


def save_evaluation_results(NON_REDUCED_FILE_PATH, REDUCED_FILE_PATH, non_reduced_metrics, reduced_metrics):
    print ('Saving GRADE evaluation results(non-reduced) into {}...'.format(
        NON_REDUCED_FILE_PATH))
    for key, value in non_reduced_metrics.items():
        if type(value) == np.ndarray:
            non_reduced_metrics[key] = value.tolist()
    write_metrics_into_json(non_reduced_metrics, NON_REDUCED_FILE_PATH)

    print ('Saving GRADE evaluation results(reduced) into {}...'.format(REDUCED_FILE_PATH))
    write_metrics_into_json(reduced_metrics, REDUCED_FILE_PATH)


def write_metrics_into_json(metrics, output_path):
    try:
        with open(output_path, 'r') as f:
            existed_metrics = json.load(f)
        existed_metrics.update(metrics)
    except IOError:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, 'w') as f:
            json.dump(existed_metrics, f, ensure_ascii=False, indent=2)


def _preprocess_adj(normalization, adj):
    adj_normalizer = fetch_normalization(normalization)
    r_adj = adj_normalizer(adj)

    return r_adj.A


def randomedge_sampler(adj, percent, normalization):
    """
    Randomly drop edge and preserve percent% edges.
    """
    adj = sp.coo_matrix(adj)
    nnz = adj.nnz #Number of nonzero matrix elements
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz*percent)
    perm = perm[:preserve_nnz]
    r_adj = sp.coo_matrix((adj.data[perm],
                            (adj.row[perm],
                            adj.col[perm])),
                            shape=adj.shape)
    r_adj = _preprocess_adj(normalization, r_adj)

    return r_adj


def load_hop_mean_embedding(oneHop_mean_path, twoHop_mean_path):
    oneHop_mean_embedding_dict = {}
    twoHop_mean_embedding_dict = {}
    with open(oneHop_mean_path, 'r') as oneHop_f, open(twoHop_mean_path, 'r') as twoHop_f:
        for line in oneHop_f.readlines():
            line_list = line.split(' ')
            word = line_list[0]
            embedding = [float(i) for i in line_list[1:]]
            oneHop_mean_embedding_dict[word] = embedding
        
        for line in twoHop_f.readlines():
            line_list = line.split(' ')
            word = line_list[0]
            embedding = [float(i) for i in line_list[1:]]
            twoHop_mean_embedding_dict[word] = embedding
    oneHop_f.close()
    twoHop_f.close()

    return oneHop_mean_embedding_dict, twoHop_mean_embedding_dict


def get_adjs1(oneHop_mean_embedding_dict, twoHop_mean_embedding_dict, input_ids_Keywords, input_ids_ctxKeywords, \
    input_ids_repKeywords, pair_hops, vocab2id, id2vocab, unlimit_hop):

    #device = input_ids_Keywords.get_device()
    device = input_ids_Keywords.device
    batch_size = input_ids_Keywords.shape[0]
    max_nodes_len = input_ids_Keywords.shape[1]
    batch_adjs = np.zeros((batch_size, max_nodes_len, max_nodes_len))

    batch_onehop_embedding_matrix = np.zeros((batch_size, max_nodes_len, 300))
    batch_twohop_embedding_matrix = np.zeros((batch_size, max_nodes_len, 300))

    # contruct keywords' embedding matrix
    for batch_id in range(batch_size):
        for index, key_id in enumerate(input_ids_Keywords[batch_id]):
            if key_id <=2:
                continue
            key_id = np.array(key_id.cpu()).tolist()
            cur_keyword = id2vocab[key_id]
            if cur_keyword in oneHop_mean_embedding_dict:
                batch_onehop_embedding_matrix[batch_id][index] = oneHop_mean_embedding_dict[cur_keyword]
            if cur_keyword in twoHop_mean_embedding_dict:
                batch_twohop_embedding_matrix[batch_id][index] = twoHop_mean_embedding_dict[cur_keyword]
 
    for batch_id in range(batch_size):
        for s_index, source_id in enumerate(input_ids_ctxKeywords[batch_id]):
            if source_id <=2: #[pad] [bos] [eos]
                continue
            source_id = np.array(source_id.cpu()).tolist()
            source = id2vocab[source_id]
            for t_index, target_id in enumerate(input_ids_repKeywords[batch_id]):
                if target_id <=2: #[bos] [eos] [pad]
                    continue

                target_id = np.array(target_id.cpu()).tolist()
                target = id2vocab[target_id]

                if (source, target) not in pair_hops: 
                    hop = unlimit_hop
                else:
                    hop = pair_hops[(source, target)]
                    if hop == -1:
                        hop = unlimit_hop

                batch_adjs[batch_id][s_index][t_index] = 1./hop 
                batch_adjs[batch_id][t_index][s_index] = 1./hop 

    for batch_id in range(batch_size):
        adj = batch_adjs[batch_id]
        tmp=adj
        adj = randomedge_sampler(adj, percent=0.80, normalization='AugNormAdj')
        batch_adjs[batch_id] = adj

    batch_adjs = torch.Tensor(batch_adjs).to(device)
    batch_onehop_embedding_matrix = torch.Tensor(batch_onehop_embedding_matrix).to(device)
    batch_twohop_embedding_matrix = torch.Tensor(batch_twohop_embedding_matrix).to(device)

    return batch_adjs, batch_onehop_embedding_matrix, batch_twohop_embedding_matrix


def get_adjs2(input_ids_Keywords, input_ids_ctxKeywords, input_ids_repKeywords, pair_hops, vocab2id, id2vocab, unlimit_hop):
    
    #device = input_ids_Keywords.get_device()
    device = input_ids_Keywords.device
    batch_size = input_ids_Keywords.shape[0]
    max_nodes_len = input_ids_Keywords.shape[1]
    batch_adjs = np.zeros((batch_size, max_nodes_len, max_nodes_len))

    for batch_id in range(batch_size):
        for s_index, source_id in enumerate(input_ids_ctxKeywords[batch_id]):
            if source_id <=2: #[pad] [bos] [eos]
                continue
            source_id = np.array(source_id.cpu()).tolist()
            source = id2vocab[source_id]
            for t_index, target_id in enumerate(input_ids_repKeywords[batch_id]):
                if target_id <=2: #[bos] [eos] [pad]
                    continue

                target_id = np.array(target_id.cpu()).tolist()
                target = id2vocab[target_id]

                if (source, target) not in pair_hops: 
                    hop = unlimit_hop
                else:
                    hop = pair_hops[(source, target)]
                    if hop == -1:
                        hop = unlimit_hop
                # print(source, target, hop)
                batch_adjs[batch_id][s_index][t_index] = 1./hop 
                batch_adjs[batch_id][t_index][s_index] = 1./hop 

    for batch_id in range(batch_size):
        adj = batch_adjs[batch_id]
        tmp=adj
        adj = randomedge_sampler(adj, percent=0.80, normalization='AugNormAdj')
        batch_adjs[batch_id] = adj

    batch_adjs = torch.Tensor(batch_adjs).to(device)

    return batch_adjs    


def load_tuples_hops(data_path):
    pair_hops = dict()
    with open(data_path, 'r') as f:
        for line in f.readlines():
            tuples = line.strip().split("|||")
            source = tuples[0]
            target = tuples[1]
            hop = int(tuples[2])

            pair_hops[(source, target)] = hop
    return pair_hops


def build_vocab_id(data_path):
    vocab2id = {}
    id2vocab = {}
    for w in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
        vocab2id[w.strip()] = len(vocab2id)
        id2vocab[len(id2vocab)] = w.strip()

    with open(data_path, 'r') as f:
        for line in f.readlines():
            w = line.strip()
            vocab2id[w] = len(vocab2id)
            id2vocab[len(id2vocab)] = w
    
    return vocab2id, id2vocab