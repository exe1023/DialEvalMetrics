import os
import csv
import logging
import json
from tqdm import tqdm
import texar.torch as tx
import numpy as np

class InputExample:
    r"""A single training/test example for simple sequence classification."""
    def __init__(self, guid, raw_text, raw_context, raw_response):
        r"""Constructs a InputExample.
        """
        self.guid = guid
        self.raw_text = raw_text
        self.raw_context = raw_context
        self.raw_response = raw_response


class InputFeatures:
    r"""A single set of features of data."""

    def __init__(self, \
        input_ids_raw_text, input_mask_raw_text, segment_ids_raw_text, \
        input_ids_raw_context, input_mask_raw_context, segment_ids_raw_context, \
        input_ids_raw_response, input_mask_raw_response, segment_ids_raw_response, \
        label_id=None):

        self.input_ids_raw_text=input_ids_raw_text,
        self.input_mask_raw_text=input_mask_raw_text,
        self.segment_ids_raw_text=segment_ids_raw_text,

        self.input_ids_raw_context=input_ids_raw_context,
        self.input_mask_raw_context=input_mask_raw_context,
        self.segment_ids_raw_context=segment_ids_raw_context,

        self.input_ids_raw_response=input_ids_raw_response,
        self.input_mask_raw_response=input_mask_raw_response,
        self.segment_ids_raw_response=segment_ids_raw_response,
        self.label_id=label_id,


class DataProcessor:
    r"""Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, pair_str, prefix):
        r"""Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, pair_str, prefix):
        r"""Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, pair_str, prefix):
        r"""Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        r"""Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv2(self, samples_dir):
        r"""Gets the list of samples."""
        samples_list = []
        with open(samples_dir, "r") as f:
            for line in f.readlines():
                line_str = line.strip()
                samples_list.append(line_str)
        return samples_list

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        r"""Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines

class DailyDialogProcessor(DataProcessor):
    r"""Processor for the DailyDialog data set."""

    def get_train_examples(self, data_dir, pair_str, prefix, model_maxlen):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv2(os.path.join(data_dir, "train", pair_str, "{}.text".format(prefix))),
            "train", model_maxlen)

    def get_dev_examples(self, data_dir, pair_str, prefix, model_maxlen):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv2(os.path.join(data_dir, "validation", pair_str, "{}.text".format(prefix))),
            "dev", model_maxlen)

    def get_test_examples(self, data_dir, pair_str, prefix, model_maxlen):
        r"""See base class."""
        return self._create_examples(
            self._read_tsv2(os.path.join(data_dir, "test", pair_str, "{}.text".format(prefix))),
            "test", model_maxlen)

    def get_labels(self):
        r"""See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(samples_list, set_type, model_maxlen):
        r"""Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(samples_list)):
            guid = f"{set_type}-{i}"
            text = ' '.join(samples_list[i].split('|||'))
            ctx = ' '.join(samples_list[i].split('|||')[:-1])
            response = samples_list[i].split('|||')[-1]

            raw_text = tx.utils.compat_as_text(text)
            raw_context = tx.utils.compat_as_text(ctx)
            raw_response = tx.utils.compat_as_text(response)

            examples.append(InputExample(guid=guid, raw_text=raw_text, \
                raw_context=raw_context, raw_response=raw_response))

        return examples

def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    r"""Converts a single `InputExample` into a single `InputFeatures`."""

    input_ids_raw_text, segment_ids_raw_text, input_mask_raw_text = \
        tokenizer.encode_text(text_a=example.raw_context,
                              text_b=example.raw_response,
                              max_seq_length=max_seq_length)
    input_ids_raw_context, segment_ids_raw_context, input_mask_raw_context = \
        tokenizer.encode_text(text_a=example.raw_context,
                              max_seq_length=max_seq_length)
    input_ids_raw_response, segment_ids_raw_response, input_mask_raw_response = \
        tokenizer.encode_text(text_a=example.raw_response,
                              max_seq_length=max_seq_length)

    # here we disable the verbose printing of the data
    if ex_index < 0:
        logging.info("*** Example ***")
        #logging.info("guid: %s", example.guid)
        logging.info("input_ids_raw_text length: %d", len(input_ids_raw_text))
        logging.info("input_ids_raw_text: %s", input_ids_raw_text)
        logging.info("input_mask_raw_text: %s", " ".join([str(x) for x in input_mask_raw_text]))
        logging.info("segment_ids_raw_text: %s", " ".join([str(x) for x in segment_ids_raw_text]))
        logging.info("input_ids_raw_context length: %d", len(input_ids_raw_context))
        logging.info("input_ids_raw_context: %s", input_ids_raw_context)
        logging.info("input_mask_raw_context: %s", " ".join([str(x) for x in input_mask_raw_context]))
        logging.info("segment_ids_raw_context: %s", " ".join([str(x) for x in segment_ids_raw_context]))
        logging.info("input_ids_raw_response length: %d", len(input_ids_raw_response))
        logging.info("input_ids_raw_response: %s", input_ids_raw_response)
        logging.info("input_mask_raw_response: %s", " ".join([str(x) for x in input_mask_raw_response]))
        logging.info("segment_ids_raw_response: %s", " ".join([str(x) for x in segment_ids_raw_response]))
        
    feature = InputFeatures(    input_ids_raw_text=input_ids_raw_text,
                                input_mask_raw_text=input_mask_raw_text,
                                segment_ids_raw_text=segment_ids_raw_text,

                                input_ids_raw_context=input_ids_raw_context,
                                input_mask_raw_context=input_mask_raw_context,
                                segment_ids_raw_context=segment_ids_raw_context,

                                input_ids_raw_response=input_ids_raw_response,
                                input_mask_raw_response=input_mask_raw_response,
                                segment_ids_raw_response=segment_ids_raw_response)
    return feature


def convert_examples_to_features_and_output_to_files(
        examples, max_seq_length, tokenizer, output_file1,
        feature_types1):
    r"""Convert a set of `InputExample`s to a pickled file."""
    with tx.data.RecordData.writer(output_file1, feature_types1) as writer1:
        for (ex_index, example) in enumerate(tqdm(examples)):
            feature = convert_single_example(ex_index, example,
                                            max_seq_length,  tokenizer)

            features = {
                "input_ids_raw_text": feature.input_ids_raw_text[0],
                "input_mask_raw_text": feature.input_mask_raw_text[0],
                "segment_ids_raw_text": feature.segment_ids_raw_text[0],

                "input_ids_raw_context": feature.input_ids_raw_context[0],
                "input_mask_raw_context": feature.input_mask_raw_context[0],
                "segment_ids_raw_context": feature.segment_ids_raw_context[0],

                "input_ids_raw_response": feature.input_ids_raw_response[0],
                "input_mask_raw_response": feature.input_mask_raw_response[0],
                "segment_ids_raw_response": feature.segment_ids_raw_response[0],
            }
            writer1.write(features)

def prepare_record_data(processor, tokenizer,
                        data_dir, pair_str, max_seq_length,  output_dir,
                        feature_types1, prefix, model_maxlen):
    r"""Prepare record data.
    """

    train_examples = processor.get_train_examples(data_dir, pair_str, prefix, model_maxlen)
    train_file1 = os.path.join(output_dir, "train", pair_str, "train_text.pkl")
    convert_examples_to_features_and_output_to_files(
        train_examples, max_seq_length,
        tokenizer, train_file1, feature_types1)

    eval_examples = processor.get_dev_examples(data_dir, pair_str, prefix, model_maxlen)
    eval_file1 = os.path.join(output_dir, "validation", pair_str, "validation_text.pkl")
    convert_examples_to_features_and_output_to_files(
        eval_examples,
        max_seq_length, tokenizer, eval_file1, feature_types1)

    test_examples = processor.get_test_examples(data_dir, pair_str, prefix, model_maxlen)
    test_file1 = os.path.join(output_dir, "test", pair_str, "test_text.pkl")
    convert_examples_to_features_and_output_to_files(
        test_examples,
        max_seq_length,  tokenizer, test_file1,feature_types1)