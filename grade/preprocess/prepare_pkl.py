import os
import sys
import argparse
import importlib
import logging
from typing import Any
import texar.torch as tx

from utils import data_utils as data_utils


OUTPUT_DIR='../data/DailyDialog'
DATA_DIR='../data/DailyDialog'

task = 'DailyDialog'
pretrained_model_name = 'bert-base-uncased'
max_seq_length = 128
max_keyword_length = 16
config_data = "config_data_grade"
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



def modify_config_data(max_seq_length, max_keyword_length, num_train_data, num_classes):
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
                        line.startswith('num_train_data =') or
                        line.startswith('max_seq_length =') or
                        line.startswith('max_keyword_length =')):
                    filedata_lines.pop(idx)
                    idx -= 1
                idx += 1

            if len(filedata_lines) > 0:
                insert_idx = 1
            else:
                insert_idx = 0
            filedata_lines.insert(
                insert_idx, f'{"num_train_data"} = {num_train_data}')
            filedata_lines.insert(
                insert_idx, f'{"num_classes"} = {num_classes}')
            filedata_lines.insert(
                insert_idx, f'{"max_seq_length"} = {max_seq_length}')
            filedata_lines.insert(
                insert_idx, f'{"max_keyword_length"} = {max_keyword_length}')

        with open("../config/{}.py".format(config_data), 'w') as file:
            file.write('\n'.join(filedata_lines))
        logging.info("{}.py has been updated".format(config_data))
    else:
        logging.info("{}.py cannot be found".format(config_data))

    logging.info("Data preparation finished")


def main():
    """ Starts the data preparation
    """
    logging.info("Loading data")

    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR
    tx.utils.maybe_create_dir(output_dir)

    processors = {
        'DailyDialog': data_utils.DailyDialogProcessor
    }
    processor = processors[task]()

    tokenizer = tx.data.BERTTokenizer(
        pretrained_model_name=pretrained_model_name)

    for pair_str in ['pair-1','pair-2']:
        num_classes = len(processor.get_labels())
        if pair_str == 'pair-1': 
            prefix = 'original_dialog' 
        else:
            prefix = 'perturbed_dialog'
        num_train_data = len(processor.get_train_examples(data_dir, pair_str, prefix, model_maxlen_for_seq))
        num_valid_data = len(processor.get_dev_examples(data_dir, pair_str, prefix, model_maxlen_for_seq))
        num_test_data = len(processor.get_test_examples(data_dir, pair_str, prefix, model_maxlen_for_seq))
        trainset_num = num_train_data
        validset_num = num_valid_data
        testset_num = num_test_data
        logging.info("num_classes: %d; num_train_data: %d, num_valid_data: %d, num_test_data: %d",
                    num_classes, num_train_data, num_valid_data, num_test_data)


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
    modify_config_data(max_seq_length, max_keyword_length, num_train_data, num_classes)


    feature_types={
        'gt_preference_label': ["int64", "stacked_tensor"]
    }
    print(trainset_num, testset_num, validset_num)
    dataset_num={
        'train':trainset_num,
        'test':testset_num,
        'validation':validset_num
    }
    for mode in ['train', 'test', 'validation']:
        output_file = '{}/{}/gt_preference_label.pkl'.format(data_dir, mode)

        with tx.data.RecordData.writer(output_file, feature_types) as writer:
            for i in range(dataset_num[mode]):
                label = 0
                feature = {
                    "gt_preference_label": label
                }
                writer.write(feature)

if __name__ == "__main__":
    main()