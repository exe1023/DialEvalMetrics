
## Requirements
List of packages we used and the version we tested the model on 

```
python == 3.7.6
pytorch_transformers  == 1.2.0
pytorch_pretrained_bert == 0.6.2
tensorboardX  == 1.9
matplotlib == 3.1.1
nltk == 3.4.5
numpy == 1.16.4
pandas == 0.25.3
scipy == 1.3.2
seaborn == 0.9.0
scikit-image == 0.15.0
torch == 1.5.0
tqdm == 4.42.1
```

### Pretrained models
Pretrained models are also available here:  
- [GPT2-finetuned-on-dailydialog](https://drive.google.com/file/d/1kNBYgIucYRBXYdCnn8CQ5mSRLYKuL4Y2/view?usp=sharing)
- [OpenNMT-dialogue-system-on-dailydialog](https://drive.google.com/file/d/1kNBYgIucYRBXYdCnn8CQ5mSRLYKuL4Y2/view?usp=sharing)

Please refer to OpenNMT website for OpenNMT model [here](https://opennmt.net/)

Pretraining the above models with commands:

**GPT2**: `--train_data_file
YOUR_FILE
--output_dir
gpt2_model/
--model_type=gpt2
--model_name_or_path=gpt2
--do_train
--do_eval
--eval_data_file
YOUR_FILE
--block_size
128
--overwrite_output_dir
--num_train_epochs
30`

**OpenNMT**:`-data
YOUR_PROCESSED_DATA
-save_model
YOUR_PATH
-layers
6
-rnn_size
512
-word_vec_size
512
-transformer_ff
2048
-heads
8
-encoder_type
transformer
-decoder_type
transformer
-position_encoding
-train_steps
100000
-max_generator_batches
2
-dropout
0.1
-batch_size
16
-batch_type
tokens
-normalization
tokens
-accum_count
2
-optim
adam
-adam_beta2
0.998
-decay_method
noam
-warmup_steps
8000
-learning_rate
2
-max_grad_norm
0
-param_init
0
-param_init_glorot
-label_smoothing
0.1
-valid_steps
2000
-save_checkpoint_steps
2000
-world_size
1
-gpu_ranks
0`


### Evaluation

Your input file should follow the format

    '''
    No header, answers are generated, and for diversity, 
    each answer is a group of sentences, separated by newlines

    | id 1 | question 1 | answer 1 | 
    | id 2 | question 2 | answer 2 |
    | id 3 | question 3 | answer 3 |
    ...


    '''

Run with arguments:

`--pretrained-model-path YOUR_PATH --metric [context | fluency | diversity | logic_consistency] --file-path YOUR_FILE.csv`


and this will append the score as another column of your .csv file
