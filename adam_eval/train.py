"""
This file is the training pipeline for the ADEM model.
"""
import argparse
import os

from models import ADEM
from preprocess import AMT_DataLoader, Preprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", type=str, help="Prototype to use (must be specified)", default='prototype_state')
    args = parser.parse_args()
    return args


def create_experiment(config):
    if not os.path.exists(config['exp_folder']):
        os.makedirs(config['exp_folder'])


if __name__ == "__main__":
    args = parse_args()
    config = eval(args.prototype)()

    print('Beginning...')
    # Set up experiment directory.
    create_experiment(config)

    # We want to make sure we have the same preprocessing across datasets.
    pp = Preprocessor()
    # This will load our training data.
    print('Loading data...')
    data_loader = AMT_DataLoader(pp, config)

    # Train our model.
    adem = ADEM(pp, config)
    print('Training...')
    adem.train_eval(data_loader, use_saved_embeddings=True)
    print('Trained!')
    adem.save()
