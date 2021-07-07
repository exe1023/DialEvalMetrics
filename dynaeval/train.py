import argparse

import torch
import os
import dgcn

log = dgcn.utils.get_logger()


def main(args):

    dgcn.utils.set_seed(args.seed)

    # load data
    log.debug("Loading data from '%s'." % args.data)
    data = dgcn.utils.load_pkl(args.data)
    log.info("Loaded data.")

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    trainset = dgcn.Dataset(data["train"], args)
    devset = dgcn.Dataset(data["dev"], args)
    testset = dgcn.Dataset(data["test"], args)

    log.debug("Building model...")
    model_file = os.path.join(args.model_save_path, args.oot_model)
    model = dgcn.DynaEval(args).to(args.device)
    opt = dgcn.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)

    coach = dgcn.Coach(trainset, devset, testset, model, opt, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)

    # Train.
    log.info("Start training...")
    coach.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--model_save_path", type=str, required=True,
                        help="Path to save model")
    # Training parameters
    parser.add_argument("--from_begin", action="store_true",
                        help="Training from begin.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Computing device.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["sgd", "rmsprop", "adam"],
                        help="Name of optimizer.")
    parser.add_argument("--learning_rate", type=float, default=0.002,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8,
                        help="Weight decay.")
    parser.add_argument("--max_grad_value", default=-1, type=float,
                        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""")
    parser.add_argument("--drop_rate", type=float, default=0.5,
                        help="Dropout rate.")


    # Model parameters
    parser.add_argument("--wp", type=int, default=10,
                        help="Past context window size. Set wp to -1 to use all the past context.")
    parser.add_argument("--wf", type=int, default=10,
                        help="Future context window size. Set wp to -1 to use all the future context.")
    parser.add_argument("--n_speakers", type=int, default=2,
                        help="Number of speakers.")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Hidden size of two layer GCN.")
    parser.add_argument("--rnn", type=str, default="lstm",
                        choices=["lstm", "gru"], help="Type of RNN cell.")
    parser.add_argument("--class_weight", action="store_true",
                        help="Use class weights in nll loss.")
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-nli-stsb-mean-tokens",
                        help="Type of bert model.")
    parser.add_argument("--sentence_dim", type=int, default=768,
                        help="dimensionality of sentence embedding")
    parser.add_argument("--max_seq_len", default=30, type=int,
                        help="maximum number of tokens per utterance.")
    parser.add_argument("--max_dialogue_len", type=int, default=110,
                        help="the longest dialogue turns")

    # others
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    parser.add_argument("--oot_model", type=str, default='epoch-02.pt',
                        help="model file name")
    args = parser.parse_args()
    log.debug(args)

    main(args)

