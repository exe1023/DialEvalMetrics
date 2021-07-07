import argparse

import torch
import os
import dgcn
from sklearn import metrics

from tqdm import tqdm

log = dgcn.utils.get_logger()


def main(args):

    dgcn.utils.set_seed(args.seed)

    # load data
    log.debug("Loading data from '%s'." % args.data)
    data = dgcn.utils.load_pkl(args.data)
    log.info("Loaded data.")

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    evalset = dgcn.Dataset(data["test"], args)

    log.debug("Building model...")
    model_file = os.path.join(args.model_save_path, args.oot_model)
    model = dgcn.DynaEval(args).to(args.device)

    ckpt = torch.load(model_file)
    best_dev_acc = ckpt["best_dev_acc"]
    best_epoch = ckpt["best_epoch"]
    best_state = ckpt["best_state"]
    model.load_state_dict(best_state)
    log.info("")
    log.info("Best epoch: {}".format(best_epoch))
    log.info("Best Dev Accuracy: {}:".format(best_dev_acc))

    model.eval()

    with torch.no_grad():
        golds = []
        preds = []
        for idx in tqdm(range(len(evalset)), desc="test"):
            data = evalset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                data[k] = v.to(args.device)
            rst = model(data)
            y_hat = rst[0]
            preds.append(y_hat.detach().to("cpu"))
        preds = torch.cat(preds, dim=-1).numpy()
        golds = torch.cat(golds, dim=-1).numpy()

        acc = metrics.accuracy_score(golds, preds)
        log.info("Test Accuracy: {}:".format(acc))

    log.info("Done evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--model_save_path", type=str, required=True,
                        help="Path to save model")
    # Training parameters
    parser.add_argument("--device", type=str, default="cpu",
                        help="Computing device.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8,
                        help="Weight decay.")
    parser.add_argument("--max_grad_value", default=-1, type=float,
                        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""")
    parser.add_argument("--drop_rate", type=float, default=0.5,
                        help="Dropout rate.")


    # Model parameters
    parser.add_argument("--wp", type=int, default=-1,
                        help="Past context window size. Set wp to -1 to use all the past context.")
    parser.add_argument("--wf", type=int, default=-1,
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
    parser.add_argument("--oot_model", type=str, default='epoch-01.pt',
                        help="model file name")
    args = parser.parse_args()
    log.debug(args)

    main(args)

