"""
This file documents the experiment parimeters to run a model.
"""


def standard_config():
    config = {}

    ## Define experiment files.

    # Reperform the preprocessing/pretraining if True. Otherwise use intermediary results.
    config['recompute'] = True
    # The folder name to store all intermediary results.
    config['exp_folder'] = '/home/ml/mnosew1/ADEM/Default'
    # The name associated with all files for the experiment.
    config['file_prefix'] = 'Default'

    ## Preprocessing.

    # The location to the raw data from amt.
    config['raw_data_location'] = '/home/ml/mnosew1/data/amt-adem/raw'

    ## Define which auxillary features to include in the model.

    features = {}
    # Calculate the BLEU scores between candidate and reference responses.
    features['bleu1'] = False
    features['bleu2'] = False
    features['bleu3'] = False
    features['bleu4'] = False
    # Return the tfidf score between the candidate and reference responses.
    features['tfidf_r'] = False
    # Return the tfidf score between the candidate response and context.
    features['tfidf_c'] = False
    # Return the mutual information score between the context and response.
    features['mutual_information'] = False
    config['feature_extraction'] = features

    ## Define which pretraining method to use.

    # Choose which model to get embeddings from.
    # Choices are 'vhred', 'dual_encoder', 'fair_autoencoder', 'tweet2vec'.
    config['pretraining'] = 'vhred'
    # True if we should reduce the dimensionality of the pretrained embeddings.
    config['use_pca'] = True
    config['pca_components'] = 50
    # True if we should oversample the data such that the candidate responses have the same distribution of scores across different lengths.
    config['oversample_length'] = True
    # For vhred, we can get the embedding from the 'CONTEXT' or 'DECODER'
    config['embedding_type'] = 'CONTEXT'
    config['vhred_bpe_file'] = '/home/ml/mnosew1/ADEM/vhred/Twitter_Codes_5000.txt'
    config['vhred_prefix'] = '/home/ml/mnosew1/ADEM/vhred/1470516214.08_TwitterModel__405001'
    config['vhred_dict'] = '/home/ml/mnosew1/ADEM/vhred/Dataset.dict.pkl'
    config['vhred_embeddings_file'] = 'vhred_embeddings.pkl'
    config['vhred_dim'] = 2000

    ## PCA parameters.

    config['pca_training_emb_file'] = 'vhred_pre_pca.pkl'
    config['raw_twitter_data'] = '/home/ml/mnosew1/data/twitter/train_twitter.txt'

    ## Model parameters.

    # True if the ADEM model should include the cMr term.
    config['use_c'] = True
    # True if the ADEM model should include the rNr' term.
    config['use_r'] = True
    # Regularization constants on the (M, N) parameters.
    config['l2_reg'] = 0.1  # 0.5
    config['l1_reg'] = 0.0
    config['bs'] = 32

    ## Training parameters.

    config['max_epochs'] = 200
    config['val_percent'] = 0.15
    config['test_percent'] = 0.15
    # Whether to validate on 'rmse' or 'pearson' correlation.
    config['validation_metric'] = 'rmse'
    # If set to a model 'hred', 'de', 'tfidf', 'human', we will leave this model out of the training set.
    config['leave_model_out'] = None

    return config


'''
This file documents the experiment parimeters to run a model.
'''


def paper_config():
    config = {}

    ## Define experiment files.

    # Reperform the preprocessing/pretraining if True. Otherwise use intermediary results.
    config['recompute'] = True
    # The folder name to store all intermediary results.
    config['exp_folder'] = './Paper'
    # The name associated with all files for the experiment.
    config['file_prefix'] = 'Paper'

    ## Preprocessing.

    # The location to the raw data from amt.
    config['raw_data_location'] = '/home/ml/mnosew1/data/amt-adem/raw'

    ## Define which pretraining method to use.

    # Choose which model to get embeddings from.
    # Choices are 'vhred', 'dual_encoder', 'fair_autoencoder', 'tweet2vec'.
    config['pretraining'] = 'vhred'
    # True if we should reduce the dimensionality of the pretrained embeddings.
    config['use_pca'] = True
    config['pca_components'] = 50
    # True if we should oversample the data such that the candidate responses have the same distribution of scores across different lengths.
    config['oversample_length'] = True
    # For vhred, we can get the embedding from the 'CONTEXT' or 'DECODER'
    config['embedding_type'] = 'CONTEXT'
    config['vhred_bpe_file'] = './vhred/Twitter_Codes_5000.txt'
    config['vhred_prefix'] = './vhred/1470516214.08_TwitterModel__405001'
    config['vhred_dict'] = './vhred/Dataset.dict.pkl'
    config['vhred_embeddings_file'] = 'vhred_embeddings.pkl'
    config['vhred_dim'] = 2000

    ## PCA parameters.

    config['pca_training_emb_file'] = 'vhred_pre_pca.pkl'
    config['raw_twitter_data'] = '/home/ml/mnosew1/data/twitter/train_twitter.txt'

    ## Model parameters.

    # True if the ADEM model should include the cMr term.
    config['use_c'] = True
    # True if the ADEM model should include the rNr' term.
    config['use_r'] = True
    # Regularization constants on the (M, N) parameters.
    config['l2_reg'] = 0.075  # 0.1#0.1#0.5
    config['l1_reg'] = 0.0
    config['bs'] = 32

    ## Training parameters.

    config['max_epochs'] = 200
    config['val_percent'] = 0.15
    config['test_percent'] = 0.15
    # Whether to validate on 'rmse' or 'pearson' correlation.
    config['validation_metric'] = 'rmse'
    # If set to a model 'hred', 'de', 'tfidf', 'human', we will leave this model out of the training set.
    config['leave_model_out'] = None

    return config


def demo_config():
    config = {}

    ## Define experiment files.

    # Reperform the preprocessing/pretraining if True. Otherwise use intermediary results.
    config['recompute'] = True
    # The folder name to store all intermediary results.
    config['exp_folder'] = './Demo'
    # The name associated with all files for the experiment.
    config['file_prefix'] = 'Demo'

    ## Preprocessing.

    # The location to the raw data from amt.
    config['raw_data_location'] = '/home/ml/mnosew1/data/amt-adem/raw'

    ## Define which pretraining method to use.

    # Choose which model to get embeddings from.
    # Choices are 'vhred', 'dual_encoder', 'fair_autoencoder', 'tweet2vec'.
    config['pretraining'] = 'vhred'
    # True if we should reduce the dimensionality of the pretrained embeddings.
    config['use_pca'] = True
    config['pca_components'] = 50
    # True if we should oversample the data such that the candidate responses have the same distribution of scores across different lengths.
    config['oversample_length'] = True
    # For vhred, we can get the embedding from the 'CONTEXT' or 'DECODER'
    config['embedding_type'] = 'CONTEXT'
    config['vhred_bpe_file'] = './vhred/Twitter_Codes_5000.txt'
    config['vhred_prefix'] = './vhred/1470516214.08_TwitterModel__405001'
    config['vhred_dict'] = './vhred/Dataset.dict.pkl'
    config['vhred_embeddings_file'] = 'vhred_embeddings.pkl'
    config['vhred_dim'] = 2000

    ## Model parameters.

    # True if the ADEM model should include the cMr term.
    config['use_c'] = True
    # True if the ADEM model should include the rNr' term.
    config['use_r'] = True
    # Regularization constants on the (M, N) parameters.
    config['l2_reg'] = 0.075
    config['l1_reg'] = 0.0
    config['bs'] = 32

    return config
