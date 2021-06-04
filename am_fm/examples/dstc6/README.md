## DSTC6 End-to-End Conversational Modeling Evaluation Task

This example is for replicating the experimental results in IWSDS 2020 Submission, 'Deep AM-FM: Toolkit For Automatic Dialogue Evaluation'.

### Dataset

1. Follow instructions at https://github.com/dialogtekgeek/DSTC6-End-to-End-Conversation-Modeling.git to collect the twitter dialogues.
2. Collect the evaluation dataset at https://www.dropbox.com/s/oh1trbos0tjzn7t/dstc6_t2_evaluation.tgz
2. Extract the training, validation and test dialogues into the data folder.

### Run Adequacy Evaluation

#### Using BERT Embedding Model (most of the steps follow the google official bert repo)

##### 1. Download the [BERT-Base, Multilingual Cased] pretrained model from https://github.com/google-research/bert and configure the BERT_BASE_DIR environment variable.

##### 2. Create preprocessed training and validation data with specific training size. This step is to conduct preprocessing on the twitter dialogues.
```bash
python ../../engines/embedding_models/bert/create_raw_data.py \
  --train_file=/path/to/train.txt \
  --train_output=/path/to/processed/train/file \
  --valid_file=/path/to/valid.txt \
  --valid_output=/path/to/processed/valid/file \
  --data_size={size of your data, such as 10000}
```

##### 3. Create tfrecord pretraining data. The tfrecord data is to easier the pretraining and faster loading. 
```bash
python ../../engines/embedding_models/bert/create_pretraining_data.py \
  --input_file=/path/to/processed/train/file \
  --output_file=/path/to/processed/train/tfrecord_file \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=60 \
  --max_predictions_per_seq=9 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

##### 4. Conduct pretraining of bert model
```bash
CUDA_VISIBLE_DEVICES=1 python ../../engines/embedding_models/bert/run_pretraining.py \
  --train_input_file=/path/to/processed/train/tfrecord_file \
  --valid_input_file=/path/to/processed/valid/tfrecord_file \
  --output_dir=/path/to/save/model \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=8 \
  --max_seq_length=60 \
  --max_predictions_per_seq=9 \
  --num_train_steps=5000 \
  --max_eval_steps=100 \
  --num_warmup_steps=100 \
  --learning_rate=2e-5
```

##### 5. Feature extraction. This step is to extract fixed word-level contextualized embedding.
```bash
CUDA_VISIBLE_DEVICES=1 python ../../engines/embedding_models/bert/extract_features.py \
  --input_file=/path/to/processed/hypothesis/file \
  --output_file=/path/to/extracted/hypothesis/json/file \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=/path/to/the/trained/checkpoint \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=60 \
  --batch_size=8
```
```bash
CUDA_VISIBLE_DEVICES=1 python ../../engines/embedding_models/bert/extract_features.py \
  --input_file=/path/to/processed/reference/file \
  --output_file=/path/to/extracted/reference/json/file \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=/path/to/the/trained/checkpoint \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=60 \
  --batch_size=8
```

##### 6. Compute AM Score
```bash
python ../../engines/embedding_models/bert/calc_am.py \
  --hyp_file=/path/to/extracted/hypothesis/json/file \
  --ref_file=/path/to/extracted/reference/json/file \
  --strategy={heuristic for form sentence embedding, such as top-layer-embedding-average}
```

### Run Fluency Evaluation

#### Using LSTM-RNN Language Model

##### 1. Follow https://github.com/google/sentencepiece.git to train a sentencepiece tokenizer with the full training set
```bash
spm_train --input=/path/to/full/processed/train/data --model_prefix=/path/to/model/prefix --vocab_size={vocabulary size} --character_coverage=0.995 --model_type=bpe
```

##### 1. Create preprocessed training and validation data (same as step 1 in Using BERT Embedding Model)

##### 2. Training the language model
```bash
SIZE={specify data size, such as 100K}  
CUDA_VISIBLE_DEVICES=3 python ../../engines/language_model/main.py \
  --data_path=/path/to/dataset \
  --dataset={name_of_dataset} \
  --data_size=${SIZE} \
  --model_name=/path/to/save/model \
  --embedding_name={name of word embedding} \
  --tokenizer_path=/path/to/sentencepiece/model \
  --hyp_out={name of hypothesis sentence level perplexity file} \
  --ref_out={name of reference sentence level perplexity file} \
  --batch_size=32 \
  --embedding_size=300 \
  --num_nodes=150,105,70 \
  --num_epochs=100 \
  --use_sp=True \
  --do_train=True \
  --do_eval=True \
  --do_dstc_eval=True
```
##### 3. Calculate FM Score
```bash
python ../../engines/language_mode/calc_fm.py \
--hyp_file=/path/to/hypothesis/perplexity/file \
--ref_file=/path/to/reference/perplexity/file
```

### Combining AM & FM

Currently, we are using weighted average to combine the system-level scores. The equation of combining them is as follow:

<p align="center">
  <img src="images/combine_am_fm.jpg"/>
</p>

### Experimental Results

#### AM BERT Model
<p align="center">
  <img src="images/table1.jpg"/>
</p>
<p align="center">
  <img src="images/table3.jpg"/>
</p>

#### FM LSTM-RNN Language Model
<p align="center">
  <img src="images/table2.jpg"/>
</p>
<p align="center">
  <img src="images/table4.jpg"/>
</p>


