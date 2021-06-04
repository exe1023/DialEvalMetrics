import codecs
import torch
import numpy as np
import math
from scipy.special import softmax
import argparse

def ngram_entropy(candsf, ngram):
    vocab = {}
    with open(candsf) as f:
        cands = [line.strip() for line in f]
    for s in cands:
        if not s:
            continue
        ws = s.split(' ')
        if len(ws)<=ngram:
            k = ' '.join(ws)
            if not k in vocab:
                vocab[k]=1
            else:
                vocab[k] = vocab[k]+1
        else:
            for i in range(len(ws)-ngram+1):
                k = ' '.join(ws[i:i+ngram])
                if not k in vocab:
                    vocab[k] = 1
                else:
                    vocab[k] = vocab[k] + 1
    total_num = sum([v for v in vocab.values()])
    print(total_num)
    print(max(vocab.values()))
    entropy = 0
    for v in vocab.values():
        entropy += -(v/total_num)*np.log((v/total_num))

    print(entropy)


def nline_ngram_entropy(candsf, outf, nline, ngram):
    wf = codecs.open(outf, 'w', encoding='utf8')
    with open(candsf) as f:
        cands = [line.strip() for line in f]
    assert len(cands)%nline==0
    for groupid in range(int(len(cands)/nline)):
        vocab = {}
        for aid in range(nline):
            s = cands[groupid*nline+aid]
            print(groupid*nline+aid)
            if not s:
                continue
            ws = s.split(' ')
            if len(ws)<=ngram:
                k = ' '.join(ws)
                if not k in vocab:
                    vocab[k]=1
                else:
                    vocab[k] = vocab[k]+1
            else:
                for i in range(len(ws)-ngram+1):
                    k = ' '.join(ws[i:i+ngram])
                    if not k in vocab:
                        vocab[k] = 1
                    else:
                        vocab[k] = vocab[k] + 1
        total_num = sum([v for v in vocab.values()])
        # print(total_num)
        # print(max(vocab.values()))
        entropy = 0
        for v in vocab.values():
            entropy += -(v/total_num)*np.log((v/total_num))

        print(entropy)
        wf.write(str(entropy)+'\n')
    wf.close()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='N-gram for each m lines')
    args_parser.add_argument('--num_line', type=int, default=5, help='Number of lines for each evaluation')
    args_parser.add_argument('--ngram', type=int, default=2, help='N-gram')
    args_parser.add_argument('--ngram_choice', choices=['nline_ngram_entropy', 'ngram_entropy'], default='nline_ngram_entropy',
                             help='Choice of N-gram')
    args_parser.add_argument('--cands_file', default='data/multipred_50000.txt', type=str,
                             help='cands_file')
    args_parser.add_argument('--output_file', default='data/nline_entropy.txt', type=str,
                             help='output_file')
    args = args_parser.parse_args()
    if args.ngram_choice is 'nline_ngram_entropy':
        nline_ngram_entropy(args.cands_file, args.output_file, args.num_line, args.ngram)
    elif args.ngram_choice is 'ngram_entropy':
        ngram_entropy(args.cands_file, args.ngram)