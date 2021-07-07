import argparse
import os
from tqdm import tqdm
import pandas as pd
import dgcn

log = dgcn.utils.get_logger()


def load_data(args):

    log.info("processing coherent dialogue data")

    train_path = os.path.join(args.data_path, "train_coherency_dset_{}.txt".format(args.perturb_type))
    dev_path = os.path.join(args.data_path, "dev_coherency_dset_{}.txt".format(args.perturb_type))
    test_path = os.path.join(args.data_path, "test_coherency_dset_{}.txt".format(args.perturb_type))

    training_data = pd.read_csv(train_path, sep='|', names=['coh_idx', 'logics1', 'utts1', 'logics2', 'utts2'])
    dev_data = pd.read_csv(dev_path, sep='|', names=['coh_idx', 'logics1', 'utts1', 'logics2', 'utts2'])
    test_data = pd.read_csv(test_path, sep='|', names=['coh_idx', 'logics1', 'utts1', 'logics2', 'utts2'])

    train, dev, test = [], [], []

    for idx in tqdm(range(len(training_data))):
        utts_1 = eval(training_data.loc[idx]['utts1'])
        utts_1 = [' '.join(item) for item in utts_1]

        utts_2 = eval(training_data.loc[idx]['utts2'])
        utts_2 = [' '.join(item) for item in utts_2]

        label = training_data.loc[idx]['coh_idx']

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

        sample = dgcn.Sample(vid="train_{}".format(idx),
                             speaker_1=spk_1_list,
                             speaker_2=spk_2_list,
                             text_1=utts_1,
                             text_2=utts_2,
                             label=label)
        train.append(sample)

    for idx in tqdm(range(len(dev_data))):
        utts_1 = eval(dev_data.loc[idx]['utts1'])
        utts_1 = [' '.join(item) for item in utts_1]

        utts_2 = eval(dev_data.loc[idx]['utts2'])
        utts_2 = [' '.join(item) for item in utts_2]

        label = dev_data.loc[idx]['coh_idx']

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

        sample = dgcn.Sample(vid="dev_{}".format(idx),
                             speaker_1=spk_1_list,
                             speaker_2=spk_2_list,
                             text_1=utts_1,
                             text_2=utts_2,
                             label=label)
        dev.append(sample)

    for idx in tqdm(range(len(test_data))):
        utts_1 = eval(test_data.loc[idx]['utts1'])
        utts_1 = [' '.join(item) for item in utts_1]

        utts_2 = eval(test_data.loc[idx]['utts2'])
        utts_2 = [' '.join(item) for item in utts_2]

        label = test_data.loc[idx]['coh_idx']

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

        sample = dgcn.Sample(vid="test_{}".format(idx),
                             speaker_1=spk_1_list,
                             speaker_2=spk_2_list,
                             text_1=utts_1,
                             text_2=utts_2,
                             label=label)
        test.append(sample)

    return train, dev, test


def main(args):
    train, dev, test = load_data(args)
    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))
    data = {"train": train, "dev": dev, "test": test}
    save_path = os.path.join(args.data_path, "{}_{}.pkl".format(args.dataset, args.perturb_type))
    dgcn.utils.save_pkl(data, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name.")
    parser.add_argument("--perturb_type", type=str, default='us',
                        help="the type pf pertubation")

    args = parser.parse_args()

    main(args)
