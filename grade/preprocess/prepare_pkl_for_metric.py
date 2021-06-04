import os
import sys
import argparse
import importlib
import logging
from typing import Any
import texar.torch as tx

from utils import data_utils_metric as data_utils


DATA_DIR = sys.argv[1]
model = sys.argv[2]
EVAL_DATASET_NAME = sys.argv[3]
HYP_FORMAT = sys.argv[4]
CTX_FORMAT = sys.argv[5]
init_embd_file = "'"+sys.argv[6]+"'"
config_data = sys.argv[7]

pickle_data_dir = "'"+DATA_DIR[1:]+"'"
task = 'DailyDialog'
pretrained_model_name = 'bert-base-uncased'
max_seq_length = 128
max_keyword_length = 16
model_maxlen_for_seq = 128


logging.root.setLevel(logging.INFO)

feature_types1={
    "input_ids_raw_text": ["int64", "stacked_tensor", max_seq_length],
    "input_mask_raw_text": ["int64", "stacked_tensor", max_seq_length],
    "segment_ids_raw_text": ["int64", "stacked_tensor", max_seq_length],

    "input_ids_raw_context": ["int64", "stacked_tensor", max_seq_length],
    "input_mask_raw_context": ["int64", "stacked_tensor", max_seq_length],
    "segment_ids_raw_context": ["int64", "stacked_tensor", max_seq_length],

    "input_ids_raw_response": ["int64", "stacked_tensor", max_seq_length],
    "input_mask_raw_response": ["int64", "stacked_tensor", max_seq_length],
    "segment_ids_raw_response": ["int64", "stacked_tensor", max_seq_length],
}



def modify_config_data(max_seq_length, max_keyword_length, num_test_data, pickle_data_dir, init_embd_file, num_classes):
    # Modify the data configuration file
    config_data_exists = os.path.isfile('../config/{}.py'.format(config_data))
    if config_data_exists:
        with open("../config/{}.py".format(config_data), 'r') as file:
            filedata = file.read()
            filedata_lines = filedata.split('\n')
            idx = 0
            while True:
                if idx >= len(filedata_lines):
                    break
                line = filedata_lines[idx]
                if (line.startswith('num_classes =') or
                        line.startswith('num_test_data =') or
                        line.startswith('max_seq_length =') or
                        line.startswith('max_keyword_length =') or
                        line.startswith('pickle_data_dir =') or
                        line.startswith('init_embd_file =')):
                    filedata_lines.pop(idx)
                    idx -= 1
                idx += 1

            if len(filedata_lines) > 0:
                insert_idx = 1
            else:
                insert_idx = 0
            filedata_lines.insert(
                insert_idx, f'{"num_test_data"} = {num_test_data}')
            filedata_lines.insert(
                insert_idx, f'{"num_classes"} = {num_classes}')
            filedata_lines.insert(
                insert_idx, f'{"max_seq_length"} = {max_seq_length}')
            filedata_lines.insert(
                insert_idx, f'{"max_keyword_length"} = {max_keyword_length}')
            filedata_lines.insert(
                insert_idx, f'{"pickle_data_dir"} = {pickle_data_dir}')
            filedata_lines.insert(
                insert_idx, f'{"init_embd_file"} = {init_embd_file}')


        with open("../config/{}.py".format(config_data), 'w') as file:
            file.write('\n'.join(filedata_lines))
        logging.info("{}.py has been updated".format(config_data))
    else:
        logging.info("{}.py cannot be found".format(config_data))

    logging.info("Data preparation finished")


def main():
    output_info = 'Prepare pkl data for evaluate and modify config_data [dialog_model: {}, dataset: {}, hyp_format: {}, ctx_format: {}]'.format(
        model, EVAL_DATASET_NAME, HYP_FORMAT, CTX_FORMAT)
    print('-' * len(output_info))
    print(output_info)
    print('-' * len(output_info))


    """ Starts the data preparation
    """
    logging.info("Loading data")

    data_dir = DATA_DIR
    output_dir = data_dir
    tx.utils.maybe_create_dir(output_dir)

    processors = {
        'DailyDialog': data_utils.DailyDialogProcessor
    }
    processor = processors[task]()

    tokenizer = tx.data.BERTTokenizer(
        pretrained_model_name=pretrained_model_name)

    # Produces pickled files
    for pair_str in ['pair-1']:
        num_classes = len(processor.get_labels())
        if pair_str == 'pair-1':
            prefix = 'original_dialog'
        else:
            prefix = 'perturbed_dialog'
        num_test_data = len(processor.get_test_examples(data_dir, pair_str, prefix, model_maxlen_for_seq))
        testset_num = num_test_data
        logging.info("num_classes: %d; num_test_data: %d",
                    num_classes, num_test_data)


        data_utils.prepare_record_data(
            processor=processor,
            tokenizer=tokenizer,
            data_dir=data_dir,
            pair_str=pair_str,
            max_seq_length=max_seq_length,
            output_dir=output_dir,
            feature_types1=feature_types1,
            prefix=prefix,
            model_maxlen=model_maxlen_for_seq)
    modify_config_data(max_seq_length, max_keyword_length, num_test_data, pickle_data_dir, init_embd_file, num_classes)


if __name__ == "__main__":
    main()
    print('Done.\n')