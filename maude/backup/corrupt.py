"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# Data corruption function used in training and evaluation
import random
import copy

class CorruptDialog():
    def __init__(self, args, data, vector_mode, bert_tokenize=False):
        self.args = args
        self.data = data
        self.vector_mode = vector_mode
        self.bert_tokenize = bert_tokenize

    def _get_response_vector(self, response):
        """
        Return the response vector
        :param response:
        :return:
        """

        if self.args.downsample:
            # set tokenize=False as we already receive tokenized words from the
            # calling methods
            return self.data.pca_predict(
                [[self.data.extract_sentence_bert(response, tokenize=False)]])[0]
        else:
            return self.data.pca_predict(response)

    def _get_token_ids(self, response):
        if self.bert_tokenize:
            return self.data.tokenizer.convert_tokens_to_ids(response)
        else:
            return [self.data.get_word_id(w) for w in response]

    def random_clean(self, dialog_id=0):
        """
        Scramble the last utterance of a dialog from a different dialog
        Can be used in mode = train and mode = test
        :param dialog_id:
        :return:
        """
        candidates = list(range(len(self.data.dialogs)))
        candidates.remove(dialog_id)
        cand_dial = random.choice(candidates)
        cand_utt = random.choice(list(range(len(self.data.dialogs[cand_dial]))))
        if self.vector_mode:
            return [self.data.dial_vecs[cand_dial][cand_utt]]
        else:
            response = self.data.dialogs[cand_dial][cand_utt]
            response = self.data.tokenizer.tokenize(response)
            return self._get_token_ids(response)

    def random_drop(self, response, drop=0.15, extract_fn=None):
        """
        Randomly drop x % of the words from the dialog
        :param drop: drop words
        :return:
        """
        # remove the special tokens
        response = self.data.tokenizer.tokenize(response)
        drop_word_pos = []
        for wi, word in enumerate(response):
            flip = random.uniform(0,1)
            if flip <= drop and word not in ['[CLS]','[SEP]']:
                drop_word_pos.append(wi)
        response = [r for i,r in enumerate(response) if i not in drop_word_pos]
        if self.vector_mode:
            return self._get_response_vector(response)
        else:
            return self._get_token_ids(response)

    def change_word_order(self, response):
        """
        Randomly shuffle the word order
        :param dialog_id:
        :return:
        """
        # remove the special tokens
        response = self.data.tokenizer.tokenize(response)
        # remove the cls from start and sep from end
        response = response[1:-1]
        # randomize the word order
        response = random.sample(response, len(response))
        response = ['[CLS]'] + response + ['[SEP]']
        if self.vector_mode:
            return self._get_response_vector(response)
        else:
            return self._get_token_ids(response)

    def get_nce_semantics(self, dialog_id=0, context_id=0):
        """

        :param dialog_id:
        :param context_id:
        :return:
        """
        if context_id % 2 == 0:
            return self.random_clean(dialog_id)
        else:
            if self.args.corrupt_type == "model_true":
                return self.get_true_model_response(dialog_id, context_id)
            elif self.args.corrupt_type == "model_false":
                return self.get_false_model_response(dialog_id, context_id)
            else:
                raise NotImplementedError("corrupt type")

    def get_true_model_response(self, dialog_id=0, context_id=0):
        """
        get the true model response
        :return:
        """
        models = self.args.corrupt_model_names.split(',')
        model = random.choice(models)
        response = self.data.model_responses[dialog_id][context_id]
        response = self.data.tokenizer.tokenize(response[model])
        response = ['[CLS]'] + response + ['[SEP]']
        if self.vector_mode:
            return self._get_response_vector(response)
        else:
            return self._get_token_ids(response)

    def get_false_model_response(self, dialog_id=0, context_id=0):
        """
        Get the corrupted model response, i.e the model response
        from a different file
        :return:
        """
        models = self.args.corrupt_model_names.split(',')
        model = random.choice(models)
        other_dials = [id for id in range(len(self.data.dialogs))]
        other_dials.remove(dialog_id)
        rand_id = random.choice(other_dials)
        c_id = random.choice(list(self.data.model_responses[rand_id].keys()))
        response = self.data.model_responses[rand_id][c_id]
        response = self.data.tokenizer.tokenize(response[model])
        response = ['[CLS]'] + response + ['[SEP]']
        if self.vector_mode:
            return self._get_response_vector(response)
        else:
            return self._get_token_ids(response)

    def get_corrupt_context_progress(self, dialog_id=0, dial_length=1):
        """
        Return a list of contexts for the current dialog, where
        the "correct" dialog in inserted in increasing order
        :return:
        """
        return list(self.next_corrupt_context_model(dialog_id, dial_length))

    def get_full_corrupt_context(self, dialog_id=0, dial_length=1):
        """
        get the corrupt context
        :param dialog_id:
        :param dial_length: depends on the length of the current dialog
        :return:
        """
        for context in self.next_corrupt_context_model(dialog_id, dial_length):
            return context

    def next_corrupt_context_model(self, dialog_id=0, dial_length=1):
        """
        start with corrupt dialog
        then iteratively append correct dialog at the bottom and remove stuff from top
        :return:
        """
        assert dial_length > 0
        candidates = list(range(len(self.data.dialogs)))
        candidates.remove(dialog_id)
        # get all dialog elements from all candidates
        current_context = self.data.dialogs[dialog_id]
        assert dial_length <= len(current_context)
        full_corrupt = []
        for utt in current_context:
            full_corrupt.append(random.choice(self.data.dialogs[random.choice(candidates)]))
        for utt_id in range(dial_length):
            corrupt_head = full_corrupt[utt_id:dial_length]
            clean_head = current_context[:utt_id]
            mixed_head = corrupt_head + clean_head
            assert len(mixed_head) == dial_length
            yield mixed_head







