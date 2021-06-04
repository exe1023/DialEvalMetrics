import argparse
import os
import random
import collections

import numpy as np
import pickle
from tqdm import tqdm
import nltk
from copy import deepcopy

from keyword_extractor import KeywordExtractor
from utils.load_data_utils import simp_tokenize, kw_tokenize


# set seed to ensure the consistency of generated data
random.seed(23)
np.random.seed(23)


class DialogDataProcessor:
    r"""Loads and pre-processes original dialog data.

    Attributes:
        dataset_name: A 'str' indicating the dataset name.
        output_data_dir: A 'str' indicating the output data directory's name.
        separator: A 'str' used to separate two utterances.
        min_kw_freq: An 'int' indicating the minimum of keyword occurrence frequency.
        context_turns: An 'int' indicating the number of turns of each dialog context.
        set_names: A 'list' of 'str' containing the set names,
            e.g., ['train', 'validation', 'test'].
    """

    def __init__(self, dataset_name, output_data_dir,
                 separator, min_kw_freq,
                 context_turns, set_names):
        self.dataset_name = dataset_name
        self.output_data_dir = output_data_dir
        self.separator = separator
        self.min_kw_freq = min_kw_freq
        self.context_turns = context_turns
        self.set_names = set_names

        self._make_data_dir_if_not_exists()
        self._load_raw_dialog_data()

        # Initializes keyword extractor
        candi_keywords = self._obtain_candidate_keywords()
        idf_dict = self._calculate_idf()
        self.kw_extractor = KeywordExtractor(candi_keywords, idf_dict)

        self._obtain_and_save_uttr_kw_mapping()  # uttr_kw_mapping: (utterances -> keywords) mapping
        self._obtain_and_save_vocab()

    def process_original_data(self):
        for name in self.set_names:
            self.current_set_name = name
            print('\nStart processing {} set...'.format(name))
            print('-' * 50)
            self._obtain_original_dialogs()
            self._extract_original_dialogs_keywords()
            self._save_processed_original_dialogs()
            print('-' * 50)

    def _make_data_dir_if_not_exists(self):
        output_data_path = '../data/{}'.format(self.output_data_dir)
        if not os.path.exists(output_data_path):
            os.makedirs(output_data_path)
        for set_name in self.set_names:
            pair1_path = os.path.join(output_data_path, set_name, 'pair-1')
            if not os.path.exists(pair1_path):
                os.makedirs(pair1_path)

    def _calculate_idf(self, load_file_if_exists=True):
        r"""Calculates and saves the IDF values for extracting keywords.

        Args:
            load_file_if_exists: A 'bool' indicating whether load IDF file if it exists.

        Returns:
            idf_dict: A 'Dict' containing all the IDF values of keywords.
        """
        if load_file_if_exists:
            idf_dict_name = '../data/{}/idf.dict'.format(self.output_data_dir)
            if os.path.isfile(idf_dict_name):
                with open(idf_dict_name, 'rb') as f:
                    idf_dict = pickle.load(f)
                print('Loading idf dict from {}'.format(idf_dict_name))
                print('idf dict size: ', len(idf_dict))
                return idf_dict

        print('Calculating idf...')

        # Calculates IDF
        counter = collections.Counter()
        total = 0.
        for dialog in tqdm(self.list_all_dialogs):
            for utterance in dialog:
                total += 1
                counter.update(set(kw_tokenize(utterance)))
        idf_dict = {}
        for k,v in counter.items():
            idf_dict[k] = np.log10(total / (v+1.))

        # Writes idf dict into file
        with open('../data/{}/idf.dict'.format(self.output_data_dir), 'wb') as f:
            pickle.dump(idf_dict, f)

        return idf_dict

    def _obtain_candidate_keywords(self, load_file_if_exists=True):
        r"""Obtains and saves the candidate keywords used for extracting keywords.

        Args:
            load_file_if_exists: A 'bool' indicating whether load candi_keywords file if it exists.

        Returns:
            candi_keywords: A 'list' containing all the candidate keywords.
        """
        if load_file_if_exists:
            candi_keywords_name = '../data/{}/candi_keywords.txt'.format(self.output_data_dir)
            if os.path.isfile(candi_keywords_name):
                with open(candi_keywords_name,'r') as f:
                    candi_keywords = [kw.strip() for kw in f.readlines()]
                print('Loading candidate keywords from {}'.format(candi_keywords_name))
                print('Total candidate keywords count: ', len(candi_keywords))
                return candi_keywords

        print('Obtaining candidate keywords...')

        # Initialization
        candi_keywords = []
        kw_counter = collections.Counter()
        kw_extractor = KeywordExtractor()

        # Extracts possible keywords.
        for dialog in tqdm(self.list_all_dialogs):
            for utterance in dialog:
                cur_keywords = kw_extractor.candi_extract(utterance)
                kw_counter.update(cur_keywords)
                candi_keywords.extend(cur_keywords)

        # Deletes the keywords occurring less than specified times
        rare_keywords = [kw for kw, freq in kw_counter.most_common()
            if freq < self.min_kw_freq]
        candi_keywords = [kw for kw, freq in kw_counter.most_common()
            if freq >= self.min_kw_freq]
        # Deletes keywords containing only one single letter
        single_letter_keywords = [kw for kw in candi_keywords if len(kw) < 2]
        candi_keywords = [kw for kw in candi_keywords if len(kw) >= 2]

        # Writes candidate keywords into file
        candidate_keywords_output_path = '../data/{}/candi_keywords.txt'.format(
            self.output_data_dir)
        with open(candidate_keywords_output_path,'w') as f:
            for keyword in candi_keywords:
                f.write(keyword + '\n')

        return candi_keywords

    def _obtain_and_save_uttr_kw_mapping(self, load_file_if_exists=True):
        r"""Obtains and saves the mapping that maps utterances into keywords they contain.

        Args:
            load_file_if_exists: A 'bool' indicating whether load mapping file if it exists.

        Returns:
            uttr_kw_mapping: A 'dict' containing utterances->keywords mapping.
        """
        if load_file_if_exists:
            uttr_kw_mapping_name = '../data/{}/uttr_kw.dict'.format(
                self.output_data_dir)
            if os.path.isfile(uttr_kw_mapping_name):
                with open(uttr_kw_mapping_name, 'rb') as f:
                    self.uttr_kw_mapping = pickle.load(f)
                print('Loading utterances->keywords mapping from {}'.format(
                    uttr_kw_mapping_name))
                print('(utterances -> keyword) mapping size: ',
                    len(self.uttr_kw_mapping))
                return

        print('Obtaining mapping from utterances to keywords...')

        # Extracts keywords to construct mapping
        self.uttr_kw_mapping = {}
        for dialog in tqdm(self.list_all_dialogs):
            for utterance in dialog:
                cur_keywords = self.kw_extractor.idf_extract(utterance)
                self.uttr_kw_mapping[utterance] = cur_keywords
        print('(utterances -> keyword) mapping size: ', len(self.uttr_kw_mapping))

        # Writes uttr_kw_mapping into file
        with open('../data/{}/uttr_kw.dict'.format(self.output_data_dir), 'wb') as f:
            pickle.dump(self.uttr_kw_mapping, f)

    def _obtain_and_save_vocab(self, load_file_if_exists=True):
        r"""Obtains and saves the vocabulary of data.
        Args:
            load_file_if_exists: A 'bool' indicating whether load vocab file if it exists.

        Returns:
            vocab: A 'list' containing all the words occurring in the data.
        """
        if load_file_if_exists:
            vocab_name = '../data/{}/vocab.txt'.format(self.output_data_dir)
            if os.path.isfile(vocab_name):
                with open(vocab_name,'r') as f:
                    self.vocab = [word.strip() for word in f.readlines()]
                print('Loading vocab from {}'.format(vocab_name))
                print('Total vocab count: ', len(self.vocab))
                return

        print('Obtain and save vocab...')

        counter = collections.Counter()
        for dialog in tqdm(self.list_all_dialogs):
            for utterance in dialog:
                counter.update(simp_tokenize(utterance))
        print('Total vocab count: ', len(counter.items()))

        # Vocab sorted by occurrence frequency (descending order)
        self.vocab = [token for token, _ in
            sorted(list(counter.items()), key=lambda x: (-x[1], x[0]))]

        # Writes vocab into file
        with open('../data/{}/vocab.txt'.format(self.output_data_dir),'w') as f:
            for word in self.vocab:
                f.write(word + '\n')

    def _load_raw_dialog_data(self):
        r"""Loads raw dialog data from files.

        Returns:
            list_all_dialogs: A 'list' containing all the dialogues, where each
                dialogue is also a 'list' containing all the utterances of this dialogue.
            dict_categorized_dialogs: a 'dict' containing the dialogue list of
                training, validation and testing set.
        """
        print('Loading raw dialog data...')
        self.list_all_dialogs = []
        self.dict_categorized_dialogs = {}
        for set_name in self.set_names:
            current_dialog_path = os.path.join(self.raw_data_dir,
                                               set_name,
                                               'dialogues_{}.txt'.format(set_name))
            with open(current_dialog_path, 'r') as f:
                raw_dialog_data = f.readlines()
            for dialog_str in tqdm(raw_dialog_data):
                dialog = self._process_dialog_str(dialog_str)
                self.list_all_dialogs.append(dialog)
                try:
                    self.dict_categorized_dialogs[set_name].append(dialog)
                except:
                    self.dict_categorized_dialogs[set_name] = [dialog]

    def _obtain_original_dialogs(self):
        # Augments the dialog data by divide each dialog into several sub-dialogs.
        print('Obtaining original dialogs...')
        self.original_dialogs = []
        for dialog in tqdm(self.list_current_dialogs):
            self.original_dialogs.extend(
                self.split_dialog(dialog, self.context_turns))

    def _extract_original_dialogs_keywords(self):
        self.original_dialogs_keywords = []
        print('Extracting keywords in original dialogs...')
        for dialog in tqdm(self.original_dialogs):
            current_dialog_keywords = []
            for utterance in dialog:
                keywords_str = ' '.join(self.uttr_kw_mapping[utterance])
                current_dialog_keywords.append(keywords_str)
            self.original_dialogs_keywords.append(current_dialog_keywords)

    def _save_processed_original_dialogs(self):
        # Saves all processed original dialog data into files
        print('Writing original dialog data into files...')
        o_text_path = os.path.join(self.current_set_output_dir,
                                   'pair-1',
                                   'original_dialog.text')
        o_kw_path = os.path.join(self.current_set_output_dir,
                                 'pair-1',
                                 'original_dialog.keyword')
        o_res_text_path = os.path.join(self.current_set_output_dir,
                                       'pair-1',
                                       'original_dialog_response.text')
        o_uni_res_text_path = os.path.join(self.current_set_output_dir,
                                       'pair-1',
                                       'original_dialog_response_uni.text')
        str_original_dialogs = self.element_to_str(self.original_dialogs, '|||')
        str_original_dialogs_keywords = self.element_to_str(
            self.original_dialogs_keywords, '|||')
        str_original_responses = [dialog[-1] for dialog in self.original_dialogs]
        uni_str_original_responses = [dialog.split('|||')[-1] for dialog in list(set(self.element_to_str(self.original_dialogs, '|||')))]
        self.save(str_original_dialogs, o_text_path)
        self.save(str_original_dialogs_keywords, o_kw_path)
        self.save(str_original_responses, o_res_text_path)
        self.save(uni_str_original_responses, o_uni_res_text_path)

    def _process_dialog_str(self, dialog_str):
        dialog = dialog_str.split(self.separator)[:-1]
        dialog = self.replace_content_in_dialog(dialog, old_content='.', new_content=' . ')
        dialog = self.replace_content_in_dialog(dialog, old_content='?', new_content=' ? ')
        dialog = self.replace_content_in_dialog(dialog, old_content=',', new_content=' , ')
        dialog = self.replace_content_in_dialog(dialog, old_content=' ’ ', new_content="'")
        dialog = [utterance.strip() for utterance in dialog]
        return dialog
# Private Methods - End
# -----------------------------------------------------------------------------

    @property
    def raw_data_dir(self):
        return './dataset/{}'.format(self.dataset_name)

    @property
    def list_current_dialogs(self):
        return self.dict_categorized_dialogs[self.current_set_name]

    @property
    def current_set_output_dir(self):
        return '../data/{}/{}/'.format(self.output_data_dir, self.current_set_name)

    @staticmethod
    def split_dialog(dialog, context_turns=1):
        r"""Split dialog into several sub-dialogs.
        Inputs: dialog, context_turns
            - **dialog**:        a 'list' containing utterances in the dialog
            - **context_turns**: how many turns of a dialogue containing in a context
        Outputs: sub_dialogs
            - **sub_dialogs**: a 'list' containing sub-dialogs
                            with respect to the current dialog

        Example:
            dialog: ['Hello!', 'Hi!', 'What's your name?', 'James.']

            assume context_turns = 1
        => (split dialog into contexts(previous utterance) and responses)

            contexts: [
                ['Hello!', 'Hi!'],
                ['Hi!', 'What's your name?'],
            ]
            responses: [
                ['What's your name?']
                ['James.']
            ]

        => (merge contexts and responses one by one)

            sub_dialogs: [
                ['Hello!', 'Hi!', 'What's your name?'],
                ['Hi!', 'What's your name?', 'James.']
            ]
        """
        num_uttr_in_context = context_turns * 2
        contexts = [
            dialog[i:i+num_uttr_in_context]
            for i in range(0, len(dialog) - num_uttr_in_context)
        ]
        responses = [[dialog[i]] for i in range(num_uttr_in_context, len(dialog))]
        sub_dialogs = [context + response for context, response in zip(contexts, responses)]
        return sub_dialogs

    @staticmethod
    def save(contents, output_path):
        with open(output_path, 'w') as f:
            for content in tqdm(contents):
                f.write(content + '\n')

    @staticmethod
    def element_to_str(contents, seperator):
        # each element in 'contents' is also a list
        return [seperator.join(element) for element in contents]

    @staticmethod
    def replace_content_in_dialog(dialog, old_content, new_content):
        r"""Replace specified content in the dialog with given new content.

        Inputs: dialog, separator
            - **dialog**:      a 'list' containing utterances in the dialog
            - **old_content**: a 'str' indicating the content needed to be replaced in the dialog
            - **new_content**: a 'str' indicating the content used to replace the old content
        Outputs: replaced_dialog
            - **replaced_dialog**: a 'list' containing all replaced utterances in the dialog

        Example:
            For an utterance ['Hello.My name is James . '],
            We wanna replace the '.' with ' . ', the procedure is as follow:
                1. first replace ' . ' with '.' obtained ['Hello.My name is James.']
                2. then replace '.' with ' . '  obtained ['Hello . My name is James . ']
        Note:
            if we replace 'old_content' with 'new_content' directly, in this example, we would get:
            ['Hello . My name is James  .  ']
        """
        # first replace the 'new_content' with 'old_content'
        # to ensure there're no utterances containing the specified 'new_content'
        replaced_dialog = [utterance.replace(new_content, old_content) for utterance in dialog]
        replaced_dialog = [utterance.replace(old_content, new_content) for utterance in replaced_dialog]
        return replaced_dialog
