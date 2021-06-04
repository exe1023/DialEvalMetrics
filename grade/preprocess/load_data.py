# Loads and pre-processes the original dialog data.

import argparse
from dialog_data_processor import DialogDataProcessor


# ==================================================================
# Configs
# ==================================================================
dataset_name = 'dailydialog'
output_data_dir = 'DailyDialog_tmp'
separator = '__eou__'  # string to separate two utterance
min_kw_freq = 1  # default minimum of keyword occurrence frequency
context_turns = 1
set_names = ['train', 'validation', 'test']


# ==================================================================
# Main
# ==================================================================
if __name__ == '__main__':
    processor = DialogDataProcessor(dataset_name, output_data_dir,
                                    separator, min_kw_freq,
                                    context_turns, set_names)
    processor.process_original_data()
    print('Done.')
