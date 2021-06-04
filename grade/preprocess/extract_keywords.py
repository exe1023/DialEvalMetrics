# Code to extract keywords

import argparse
import os
import pickle
import collections
from tqdm import tqdm
import numpy as np
import json

from keyword_extractor import KeywordExtractor
from utils.load_data_utils import kw_tokenize


# ==================================================================
# Function Definition
# ==================================================================
def _calculate_idf(list_all_dialogs, idf_path=None, load_file_if_exists=True):
    if load_file_if_exists:
        if os.path.isfile(idf_path):
            with open(idf_path, 'rb') as f:
                idf_dict = pickle.load(f)
            print('Loading idf dict from {}'.format(idf_path))
            print('idf dict size: ', len(idf_dict))
            return idf_dict

    if not list_all_dialogs:
        raise Exception('no dialogs provided for calculating IDF')

    idf_dir = os.path.dirname(idf_path)
    if not os.path.exists(idf_dir):
        os.makedirs(idf_dir)

    print('Calculating idf...')
    # calculate IDF
    counter = collections.Counter()
    total = 0.
    for dialog in tqdm(list_all_dialogs):
        for utterance in dialog:
            total += 1
            counter.update(set(kw_tokenize(utterance)))
    idf_dict = {}
    for k,v in counter.items():
        idf_dict[k] = np.log10(total / (v+1.))

    print('Saving idf into {}...'.format(idf_path))
    with open(idf_path, 'wb') as f:
        pickle.dump(idf_dict, f)

    return idf_dict


def _obtain_candidate_keywords(list_all_dialogs, candi_kw_path, min_kw_freq=1, load_file_if_exists=True):
    r"""Obtain and save the candidate keywords used for extracting keywords.

    Inputs: list_all_dialogs, candi_kw_path, load_file_if_exists
        # TODO
        - **list_all_dialogs**:
        - **candi_kw_path**:
        - **load_file_if_exists**:

    Outputs: candi_keywords
        - **candi_keywords**:  a 'list' containing all the candidate keywords
    """
    if load_file_if_exists:
        if os.path.isfile(candi_kw_path):
            with open(candi_kw_path,'r') as f:
                candi_keywords = [kw.strip() for kw in f.readlines()]
            print('Loading candidate keywords from {}'.format(candi_kw_path))
            print('Total candidate keywords count: ', len(candi_keywords))
            return candi_keywords

    if not list_all_dialogs:
        raise Exception('no dialogs provided for obtaining candidate keywords')

    candi_kw_dir = os.path.dirname(candi_kw_path)
    if not os.path.exists(candi_kw_dir):
        os.makedirs(candi_kw_dir)

    print('Obtaining candidate keywords...')

    # initialization
    candi_keywords = []
    kw_counter = collections.Counter()
    kw_extractor = KeywordExtractor()

    # extract possible keywords
    for dialog in tqdm(list_all_dialogs):
        for utterance in dialog:
            cur_keywords = kw_extractor.candi_extract(utterance)
            kw_counter.update(cur_keywords)
            candi_keywords.extend(cur_keywords)

    # delete the keywords occurring less than specified times (indicated by 'min_kw_freq').
    rare_keywords = [kw for kw, freq in kw_counter.most_common() if freq < min_kw_freq]
    candi_keywords = [kw for kw, freq in kw_counter.most_common() if freq >= min_kw_freq]
    # delete keywords containing only one single letter
    single_letter_keywords = [kw for kw in candi_keywords if len(kw) < 2]
    candi_keywords = [kw for kw in candi_keywords if len(kw) >= 2]

    # print the information of candidate keywords
    print('rare keywords count: ', len(rare_keywords))
    print('single letter keywords count: ', len(single_letter_keywords))
    print('total candidate keywords count(before cleaning): ', len(kw_counter.items()))
    print('total candidate keywords count(after cleaning):  ', len(candi_keywords))

    print('Saving candi_keywords into {}...'.format(candi_kw_path))
    with open(candi_kw_path,'w') as f:
        for keyword in candi_keywords:
            f.write(keyword + '\n')

    return candi_keywords


def load_dataset(dataset_name, dataset_dir):
    if dataset_name == 'dailydialog':
        data = load_dailydialog(dataset_dir)
    if dataset_name == 'convai2':
        data = load_convai2(dataset_dir)
    if dataset_name == 'empatheticdialogues':
        data = load_empatheticdialogues(dataset_dir)
    if 'eval' in dataset_name:
        data = load_evaldata(dataset_dir)
    return data

def load_evaldata(dataset_dir):
    with open(f'{dataset_dir}/data.json') as f:
        dialog_data = json.load(f)
    return dialog_data

def load_dailydialog(dataset_dir, separator='__eou__'):
    print('Loading raw dialog data of DailyDialog...')
    dataset_path = os.path.join(dataset_dir, 'dialogues_text.txt')
    with open(dataset_path, 'r') as f:
        dialog_data = [process_dialog_str(dialog_str, separator) for dialog_str in f.readlines()]
    return dialog_data


def load_convai2(dataset_dir):
    print('Loading raw dialog data of convai2...')
    def pop_one_sample(lines):
        dialog = []

        started = False
        while len(lines) > 0:
            line = lines.pop()
            id, context = line.split(' ', 1)
            id = int(id)
            context = context.strip()

            if started == False: # not started
                assert id == 1
                started = True
            elif id == 1: # break for next
                lines.append(line)
                break

            try:
                uttr, response = context.split('\t', 2)[:2]
                dialog.append(uttr)
                dialog.append(response)
            except:
                uttr = context
                dialog.append(uttr)

        return dialog

    dataset_path = os.path.join(dataset_dir, 'convai2_original.txt')
    with open(dataset_path, 'r') as f:
        lines = f.readlines()[::-1]
    dialog_data = []
    while len(lines) > 0:
        dialog_data.append(pop_one_sample(lines))
    return dialog_data


def load_empatheticdialogues(dataset_dir):
    print('Loading raw dialog data of empathetic dialogues...')
    dataset_path = os.path.join(dataset_dir, 'all.csv')
    with open(dataset_path) as f:
        df = f.readlines()
    dialog_data = []
    dialog = []
    for i in range(1, len(df)):

        cparts = df[i - 1].strip().split(",")
        sparts = df[i].strip().split(",")

        if cparts[0] == sparts[0]:
            contextt = cparts[5].replace("_comma_", ",").strip()
            dialog.append(contextt)
        else:
            if len(dialog) > 0:
                dialog_data.append(dialog)
            dialog = []

    return dialog_data


def process_dialog_str(dialog_str, separator):
    dialog = dialog_str.split(separator)[:-1]
    dialog = replace_content_in_dialog(dialog, old_content='.', new_content=' . ')
    dialog = replace_content_in_dialog(dialog, old_content='?', new_content=' ? ')
    dialog = replace_content_in_dialog(dialog, old_content=',', new_content=' , ')
    dialog = replace_content_in_dialog(dialog, old_content=' ’ ', new_content="'")
    dialog = [utterance.strip() for utterance in dialog]
    return dialog


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


def load_texts(text_input_path, seperator='|||'):
    with open(text_input_path, 'r') as f:
        texts = [line.strip().split(seperator) for line in f.readlines()]
    return texts


def extract_keywords(dialogs, kw_extractor, kw_output_path,  seperator='|||'):
    dialogs_keywords = [[' '.join(kw_extractor.idf_extract(utterance)) for utterance in dialog]
                        for dialog in tqdm(dialogs)]
    with open(kw_output_path, 'w') as f:
        print('Saving keywords into {}...'.format(kw_output_path))
        for keywords_in_a_dialog in dialogs_keywords:
            f.write(seperator.join(keywords_in_a_dialog) + '\n')


# ==================================================================
# Main
# ==================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='name of dataset')
    parser.add_argument('--dataset_dir', type=str, help='path of dataset file')
    parser.add_argument('--idf_path', type=str, help='path of idf file')
    parser.add_argument('--candi_kw_path', type=str, help='path of candidate keywords file')
    parser.add_argument('--input_text_path', type=str, help='path of dialog text that need extracting keywords')
    parser.add_argument('--kw_output_path', type=str, help='path of dialog text that need extracting keywords')
    args = parser.parse_args()

    output_info = 'Start keyword extraction [dataset: {}, file: {}]'.format(
        args.dataset_name, args.input_text_path)
    print('-' * len(output_info))
    print(output_info)
    print('-' * len(output_info))

    # initialize keyword extractor
    try:
        candi_keywords = _obtain_candidate_keywords(None, args.candi_kw_path)
        idf_dict = _calculate_idf(None, args.idf_path)
        kw_extractor = KeywordExtractor(candi_keywords, idf_dict)
    except Exception as err:
        print('Exception: ', err)
        # load all dialogs of the specific dataset
        dataset = load_dataset(args.dataset_name, args.dataset_dir)
        candi_keywords = _obtain_candidate_keywords(dataset, args.candi_kw_path)
        idf_dict = _calculate_idf(dataset, args.idf_path)
        kw_extractor = KeywordExtractor(candi_keywords, idf_dict)


    # load texts that need extracting keywords
    texts = load_texts(args.input_text_path)
    # extract keywords
    extract_keywords(texts, kw_extractor, args.kw_output_path)
    print('Done.')
