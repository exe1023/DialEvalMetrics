import copy
max_train_bert_epoch = 5
num_train_data = 118528
pickle_data_dir = './data/DailyDialog'
train_batch_size = 16
init_embd_file = './tools/numberbatch-en-19.08.txt'
max_keyword_length = 16 
max_seq_length = 128
num_classes = 2

vocab_file = '{}/keyword.vocab'.format(pickle_data_dir)
display_steps = 1000 # Print training loss every display_steps; -1 to disable
save_steps = -1
eval_steps = 5000 # Eval on the dev set every eval_steps; -1 to disable
# Proportion of training to perform linear learning rate warmup for.
# E.g., 0.1 = 10% of training.
warmup_proportion = 0.1
eval_batch_size = 16
test_batch_size = 16

metric_pickle_data_dir = './data/DailyDialog/daily_metric'

feature_types = {
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

metricData_feature_types = {
    "input_ids_raw_text": ["int64", "stacked_tensor", max_seq_length],
    "input_mask_raw_text": ["int64", "stacked_tensor", max_seq_length],
    "segment_ids_raw_text": ["int64", "stacked_tensor", max_seq_length]
}



train_hparam = {
    "allow_smaller_final_batch": False,
    "batch_size": train_batch_size,
    "datasets": [

        ############################# pair-1 #############################
        {
            "files": "{}/train/pair-1/train_text.pkl".format(pickle_data_dir),
            'data_name': 'pair_1',
            'data_type': 'record',
            "feature_types": feature_types,
        },
        {
            "files": "{}/train/pair-1/original_dialog_merge.keyword".format(pickle_data_dir),
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
            "files": "{}/train/pair-1/original_dialog_merge.ctx_keyword".format(pickle_data_dir),
            'data_name': 'ctx_keyword_pair_1',
            "vocab_share_with":1,
            "embedding_init_share_with":1,
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        },
        {
            "files": "{}/train/pair-1/original_dialog_merge.rep_keyword".format(pickle_data_dir),
            'data_name': 'rep_keyword_pair_1',
            "vocab_share_with":1,
            "embedding_init_share_with":1,
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        },


        ############################# pair-2 #############################
        {
            "files": "{}/train/pair-2/train_text.pkl".format(pickle_data_dir),
            'data_name': 'pair_2',
            'data_type': 'record',
            "feature_types": feature_types,
        },
        {
            "files": "{}/train/pair-2/perturbed_dialog_merge.keyword".format(pickle_data_dir),
            'data_name': 'keyword_pair_2',
            "vocab_share_with":1,
            "embedding_init_share_with":1,
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        },
        {
            "files": "{}/train/pair-2/perturbed_dialog_merge.ctx_keyword".format(pickle_data_dir),
            'data_name': 'ctx_keyword_pair_2',
            "vocab_share_with":1,
            "embedding_init_share_with":1,
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        },
        {
            "files": "{}/train/pair-2/perturbed_dialog_merge.rep_keyword".format(pickle_data_dir),
            'data_name': 'rep_keyword_pair_2',
            "vocab_share_with":1,
            "embedding_init_share_with":1,
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        },


        {
            "files": "{}/train/gt_preference_label.pkl".format(pickle_data_dir),
            'data_type': 'record',
            "feature_types": {
                'gt_preference_label': ["int64", "stacked_tensor"]
            },
        }
    ],
    "shuffle": True
}


eval_hparam = copy.deepcopy(train_hparam)
eval_hparam['allow_smaller_final_batch'] = True
eval_hparam['batch_size'] = eval_batch_size
eval_hparam['shuffle'] = False
eval_hparam['datasets'][0]['files'] = "{}/validation/pair-1/validation_text.pkl".format(pickle_data_dir)
eval_hparam['datasets'][1]['files'] = "{}/validation/pair-1/original_dialog_merge.keyword".format(pickle_data_dir)
eval_hparam['datasets'][2]['files'] = "{}/validation/pair-1/original_dialog_merge.ctx_keyword".format(pickle_data_dir)
eval_hparam['datasets'][3]['files'] = "{}/validation/pair-1/original_dialog_merge.rep_keyword".format(pickle_data_dir)
eval_hparam['datasets'][4]['files'] = "{}/validation/pair-2/validation_text.pkl".format(pickle_data_dir)
eval_hparam['datasets'][5]['files'] = "{}/validation/pair-2/perturbed_dialog_merge.keyword".format(pickle_data_dir)
eval_hparam['datasets'][6]['files'] = "{}/validation/pair-2/perturbed_dialog_merge.ctx_keyword".format(pickle_data_dir)
eval_hparam['datasets'][7]['files'] = "{}/validation/pair-2/perturbed_dialog_merge.rep_keyword".format(pickle_data_dir)
eval_hparam['datasets'][8]['files'] = "{}/validation/gt_preference_label.pkl".format(pickle_data_dir)


test_hparam = copy.deepcopy(train_hparam)
test_hparam['allow_smaller_final_batch'] = True
test_hparam['batch_size'] = test_batch_size
test_hparam['shuffle'] = False
test_hparam['datasets'][0]['files'] = "{}/test/pair-1/test_text.pkl".format(pickle_data_dir)
test_hparam['datasets'][1]['files'] = "{}/test/pair-1/original_dialog_merge.keyword".format(pickle_data_dir)
test_hparam['datasets'][2]['files'] = "{}/test/pair-1/original_dialog_merge.ctx_keyword".format(pickle_data_dir)
test_hparam['datasets'][3]['files'] = "{}/test/pair-1/original_dialog_merge.rep_keyword".format(pickle_data_dir)
test_hparam['datasets'][4]['files'] = "{}/test/pair-2/test_text.pkl".format(pickle_data_dir)
test_hparam['datasets'][5]['files'] = "{}/test/pair-2/perturbed_dialog_merge.keyword".format(pickle_data_dir)
test_hparam['datasets'][6]['files'] = "{}/test/pair-2/perturbed_dialog_merge.ctx_keyword".format(pickle_data_dir)
test_hparam['datasets'][7]['files'] = "{}/test/pair-2/perturbed_dialog_merge.rep_keyword".format(pickle_data_dir)
test_hparam['datasets'][8]['files'] = "{}/test/gt_preference_label.pkl".format(pickle_data_dir)


metric_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "datasets": [
        {
            "files": "{}/dialog.pkl".format(metric_pickle_data_dir),
            'data_name': 'metric',
            'data_type': 'record',
            "feature_types": metricData_feature_types,
        },
        {
            "files": "{}/dialog_merge.keyword".format(metric_pickle_data_dir),
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
            "files": "{}/dialog_merge.ctx_keyword".format(metric_pickle_data_dir),
            'data_name': 'ctx_keyword_pair_1',
            "vocab_share_with":1,
            "embedding_init_share_with":1,
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        },
        {
            "files": "{}/dialog_merge.rep_keyword".format(metric_pickle_data_dir),
            'data_name': 'rep_keyword_pair_1',
            "vocab_share_with":1,
            "embedding_init_share_with":1,
            "max_seq_length": max_keyword_length, #The length does not include any added "bos_token" or "eos_token"
        },
    ],
    "shuffle": False
}