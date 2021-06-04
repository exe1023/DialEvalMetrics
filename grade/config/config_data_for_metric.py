import copy
init_embd_file = './tools/numberbatch-en-19.08.txt'
pickle_data_dir = './data/eval_engage'
max_keyword_length = 16
max_seq_length = 128
num_classes = 2
num_test_data = 600

vocab_file = './data/DailyDialog/keyword.vocab'
train_batch_size = 8
max_train_epoch = 20
pretrained_epoch = -1
display_steps = 50  # Print training loss every display_steps; -1 to disable


eval_steps = 100  # Eval on the dev set every eval_steps; -1 to disable
# Proportion of training to perform linear learning rate warmup for.
# E.g., 0.1 = 10% of training.
warmup_proportion = 0.1
eval_batch_size = 32
test_batch_size = 32


feature_types = {
    # Reading features from pickled data file.
    # E.g., Reading feature "input_ids" as dtype `int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
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


test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "datasets": [
        {
            "files": "{}/test/pair-1/test_text.pkl".format(pickle_data_dir),
            'data_name': 'pair_1',
            'data_type': 'record',
            "feature_types": feature_types,
        },
        {
            "files": "{}/test/pair-1/original_dialog_merge.keyword".format(pickle_data_dir),
            'data_name': 'keyword_pair_1',
            'vocab_file': vocab_file,
            "embedding_init": {
                "file": init_embd_file,
                'dim':300,
                'read_fn':"load_glove"
            },
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        },
        {
            "files": "{}/test/pair-1/original_dialog_merge.ctx_keyword".format(pickle_data_dir),
            'data_name': 'ctx_keyword_pair_1',
            "vocab_share_with":1,
            "embedding_init_share_with":1,
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        },
        {
            "files": "{}/test/pair-1/original_dialog_merge.rep_keyword".format(pickle_data_dir),
            'data_name': 'rep_keyword_pair_1',
            "vocab_share_with":1,
            "embedding_init_share_with":1,
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        }
    ],
    "shuffle": False,
}
