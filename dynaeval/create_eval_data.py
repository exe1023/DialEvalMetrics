import argparse
import os
from tqdm import tqdm
import pandas as pd
import dgcn
import json

log = dgcn.utils.get_logger()


def load_data(args):

    log.info("processing coherent dialogue data")

    #eval_path = os.path.join(args.data_path, "coherency_dset_{}.txt".format(args.dataset))
    eval_path = os.path.join(args.data_path, "data.csv".format(args.dataset))

    eval_data = pd.read_csv(eval_path, sep='|', names=['coh_idx', 'logics1', 'utts1', 'logics2', 'utts2'])

    eval_list = []

    for idx in tqdm(range(len(eval_data))):
        utts_1 = eval(eval_data.loc[idx]['utts1'])
        #utts_1 = [' '.join(item) for item in utts_1]

        utts_2 = eval(eval_data.loc[idx]['utts2'])
        #utts_2 = [' '.join(item) for item in utts_2]

        label = eval_data.loc[idx]['coh_idx']

        spk_1_list = []
        spk_2_list = []

        for j in range(len(utts_1)):
            if j % 2 == 0:
                spk_1_list.append('A')
            else:
                spk_1_list.append('B')

        for j in range(len(utts_2)):
            if j % 2 == 0:
                spk_2_list.append('A')
            else:
                spk_2_list.append('B')

        sample = dgcn.Sample(vid="eval_{}".format(idx),
                             speaker_1=spk_1_list,
                             speaker_2=spk_2_list,
                             text_1=utts_1,
                             text_2=utts_2,
                             label=label)
        eval_list.append(sample)

    return eval_list


def main(args):
    eval_list = load_data(args)
    log.info("number of train samples: {}".format(len(eval_list)))
    data = {"eval": eval_list}
    save_path = os.path.join(args.data_path, "{}_eval.pkl".format(args.dataset))
    dgcn.utils.save_pkl(data, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name.")
    args = parser.parse_args()

    main(args)
