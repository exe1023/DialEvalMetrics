''' The methods in this file will take a dataset in and embed each context, response,
	and model response.

	Each method should oversample according to length and perform pca if specified in
	config.
'''
import math
import pickle

import numpy as np

from adam_eval.apply_bpe import BPE
from adam_eval.vhred_py.vhred_compute_dialogue_embeddings import compute_encodings as VHRED_compute_encodings
from adam_eval.vhred_py.vhred_dialog_encdec import DialogEncoderDecoder as VHRED_DialogEncoderDecoder
from adam_eval.vhred_py.vhred_state import prototype_state as VHRED_prototype_state

np.random.seed(0)


class VHRED(object):
    def __init__(self, config):
        self.config = config
        self.f_dict = config['vhred_dict']
        # Load the VHRED model.
        self.model, self.enc_fn, self.dec_fn = self._build_vhred_model()
        # Load in Twitter dictionaries for BPE conversion.
        f_bpe_dictionary = config['vhred_bpe_file']
        with open(f_bpe_dictionary, 'r') as handle:
            self.bpe = BPE(handle.readlines(), '@@')
        with open(self.f_dict, 'rb') as handle:
            twitter_dict = pickle.load(handle)
        self.str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in twitter_dict])
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id, _, _ in twitter_dict])
        self.MODELS = ['hred', 'human', 'tfidf', 'de']

    def _convert_text_to_bpe(self, contexts, gt_responses, model_responses, ignore_models=False):
        # Files needed for BPE conversions.
        context_ids = self._strs_to_idxs(contexts)
        gt_response_ids = self._strs_to_idxs(gt_responses)

        longest = 0
        for res in gt_response_ids:
            if len(res) > longest:
                longest = len(res)
        print('Longest Response:', longest)

        if not ignore_models:
            model_response_ids = self._strs_to_idxs(model_responses)
        else:
            model_response_ids = None
        return context_ids, gt_response_ids, model_response_ids

    def _strs_to_idxs(self, data):
        out = []
        for row in data:
            bpe_segmented = self.bpe.segment(row.strip())
            out.append([self.str_to_idx[word] for word in bpe_segmented.split() if word in self.str_to_idx])
        return out

    def _idxs_to_strs(self, data):
        out = []
        for row in data:
            s = ' '.join([self.idx_to_str[word] for word in row])
            out.append(s.replace('@@ ', ''))
        return out

    def _build_vhred_model(self):
        # Update the state dictionary.
        state = VHRED_prototype_state()
        model_prefix = self.config['vhred_prefix']
        state_path = model_prefix + "_state.pkl"
        model_path = model_prefix + "_model.npz"
        with open(state_path, 'rb') as handle:
            state.update(pickle.load(handle))
        # Update the bs for the current data.
        state['bs'] = 100
        state['dictionary'] = self.f_dict

        # Create the model:
        model = VHRED_DialogEncoderDecoder(state)
        # Load model weights.
        model.load(model_path)
        model.bs = 100
        enc_fn = model.build_encoder_function()
        dec_fn = model.build_decoder_encoding()

        return model, enc_fn, dec_fn

    def _extract_text(self, dataset, ignore_models=False):
        cs, gt_rs, m_rs = [], [], []
        for entry in dataset:
            cs.append(entry['c'])
            gt_rs.append(entry['r_gt'])
            # Extract in this order so we don't mix up which responses came from which models.
            if not ignore_models:
                for m_name in self.MODELS:
                    m_rs.append(entry['r_models'][m_name][0])

        # Add </s> token to beginning of each.
        cs = ['</s> ' + c.strip() if '</s> ' not in c[0:6] else c.strip() for c in cs]
        gt_rs = ['</s> ' + c.strip() if '</s> ' not in c[0:6] else c.strip() for c in gt_rs]
        if not ignore_models:
            m_rs = ['</s> ' + c.strip() if '</s> ' not in c[0:6] else c.strip() for c in m_rs]

        return cs, gt_rs, m_rs

    # Compute model embeddings for contexts or responses
    # Embedding type can be 'CONTEXT' or 'DECODER'
    def _compute_embeddings(self, data):
        embeddings = []
        context_ids_batch = []
        batch_index = 0
        batch_total = int(math.ceil(float(len(data)) / float(self.model.bs)))

        counter = 0
        max_len = 0
        for context_ids in data:
            counter += 1
            context_ids_batch.append(context_ids)

            # If we have filled up a batch, or reached the end of our data:
            if len(context_ids_batch) == self.model.bs or counter == len(data):
                batch_index += 1
                length = len(context_ids_batch)
                if len(context_ids_batch) < self.model.bs:
                    # Pad the data to get a full batch.
                    while len(context_ids_batch) < self.model.bs:
                        context_ids_batch.append(context_ids_batch[0])
                print('Computing embeddings for batch %d/%d' % (batch_index, batch_total))
                encs = VHRED_compute_encodings(context_ids_batch, self.model, self.enc_fn, self.dec_fn,
                                               self.config['embedding_type'])
                if length < self.model.bs:
                    encs = encs[:length]
                for i in range(len(encs)):
                    embeddings.append(encs[i, :].tolist())
                context_ids_batch = []

        return embeddings

    def _add_embeddings_to_dataset(self, dataset, c_embs, r_gt_embs, r_model_embs, ignore_models=False):
        for ix in range(len(dataset)):
            dataset[ix]['c_emb'] = c_embs[ix]
            dataset[ix]['r_gt_emb'] = r_gt_embs[ix]
            if not ignore_models:
                dataset[ix]['r_model_embs'] = {}
                for jx, m_name in enumerate(self.MODELS):
                    dataset[ix]['r_model_embs'][m_name] = r_model_embs[ix * len(self.MODELS) + jx]
        return dataset

    def get_embeddings(self, dataset, new_models=None, ignore_models=False):
        """
        Dataset should be a list of dictionaries. Each dictionary should have
        keys: c, r_gt, r_models = {'model_name': [r, score, length], ...}
        """
        if not new_models is None:
            self.MODELS = new_models
        if 'r_models' not in dataset[0]:
            ignore_models = True

        contexts, gt_responses, model_responses = self._extract_text(dataset, ignore_models=ignore_models)
        context_ids, gt_response_ids, model_response_ids = self._convert_text_to_bpe(contexts, gt_responses,
                                                                                     model_responses,
                                                                                     ignore_models=ignore_models)

        print('Computing context embeddings...')
        context_embs = self._compute_embeddings(context_ids)
        print('Computing ground truth response embeddings...')
        gt_response_embs = self._compute_embeddings(gt_response_ids)

        if not ignore_models:
            print('Computing model response embeddings...')
            model_response_embs = self._compute_embeddings(model_response_ids)
        else:
            model_response_embs = None

        # Update our dataset with each of the embeddings.
        dataset = self._add_embeddings_to_dataset(dataset, context_embs, gt_response_embs, model_response_embs,
                                                  ignore_models=ignore_models)

        return dataset

    def use_saved_embeddings(self):
        with open(self.config['vhred_embeddings_file'], 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset
