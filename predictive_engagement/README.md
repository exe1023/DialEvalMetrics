# PredictiveEngagement

This repository contains code for [Predictive Engagement](https://arxiv.org/pdf/1911.01456.pdf) paper. If you use it please cite it as: @inproceedings{Ghazarian2020, title={Predictive Engagement: An Efficient Metric For Automatic Evaluation of
Open-Domain Dialogue Systems}, author={Sarik Ghazarian, Ralph Weischedel, Aram Galstyan and Nanyun Peng}, year={2020} }



For any comments/issues/ideas pleaae feel free to contact [me](mailto:sarikgha@usc.edu).


## Steps to setup

### Install Requirements
Use any virtualenv manager to install all the packages mentioned in the requirements.txt file.

In order to train/test engagement classifier or predict the engagment scores based on trained models please follow these steps:

### 1. Preprocess dataset
Run the preprocess.py in the pytorch_src directory. This script preprocesses ConvAI dataset (train.json file taken from http://convai.io/2017/data/) to extract the dialogs with at least one turn (query and reply utterances). This extracted files are needed to train and test the engagement classifier.
The outputs are:
* ConvAI_convs_orig : includes all 2099 conversations from ConvAI dataset with at least one turn of dialog
* ConvAI_convs : includes all conversations from ConvAI except the 50 dialogs used for AMT experiments (Table 2. of paper)
* ConvAI_convs_train, ConvAI_convs_test, ConvAI_convs_valid: include 60/20/20 percent of conversations from ConvAI dataset as train/test/valid sets
* ConvAI_utts_train.csv, ConvAI_utts_test.csv, ConvAI_utts_valid.csv: train/test/valid sets of utterances from ConvAI dataset containing queries, replies and their corresponding engagement labels used for utterance-level engagement classifier.


### 2. Utterance embeddings
In order to train the engagement classifier or test the trained model, you need to have a set of embeddings files for queries and replies, where each utterance embedding is the mean or max pooling of its words embeddings. In this paper, we have used the Bert contextualized embeddings.
Run create_utt_embed.py with queries and replies files as input to create their embeddings by using BertClient and BertServer 
(cite: @misc{xiao2018bertservice,title={bert-as-service},author={Xiao, Han},howpublished={\url{https://github.com/hanxiao/bert-as-service}},ear={2018}}).
According to https://github.com/hanxiao/bert-as-service, before running create_utt_embed.py which serves as BertClient, you need to start BertServer with following command where model_dir in this command is the directory that pretrained Bert model has been downloaded in:

bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=128 -pooling_strategy=REDUCE_MEAN


### 3. Train/test and finetune model
Run main.py in order to train the engagement classifier, test it, finetune it on Daily Dilaog dataset and predict engagement scores for queries and replies of Daily Dialoge test set.
Model directory includes the engagement classifier trained on ConvAI dataset (model/best_model.pt) and finetuned on Daily Dialog set (model/best_model_finetuned.pt). The both models are based on mean pooling of word embeddings.
cd into pytorch_src/ directory and specify the mode and all the parameter values that you need to run. 

To train the model from scratch on ConvAI data:   python main.py --mode train --pooling mean 
(The best model (model/best_model.pt) is an MLP classifier with [64,32,8] hiddent units, lr=0.001, dropout=0.8, regularizer=0.001, Adam)

To test the trained model on ConvAI test set:   python main.py --mode test --pooling mean 

To test the trained model on 297 utterances (Table 2. in paper) annotated by AMT workers:   python main.py --mode testAMT --pooling mean

To finetune the trained model on DailyDilaog datset:   python main.py --mode finetune --pooling mean 

To predict the engagement score based on finetuned model:   python main.py --mode predict --pooling mean 


### 4. Pearson and Spearman Correlations 
Run calculate_correlations.py to compute the Pearson and Spearman correlations between human annotations and engagement scores predicted by different models.



