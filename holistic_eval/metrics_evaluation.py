#from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import pandas as pd
import argparse
from roberta_mnli.logic_eval_interface import logic_eval

def fluency_score(rated_a, opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(opt.pretrained_model_path)
    model = GPT2LMHeadModel.from_pretrained(opt.pretrained_model_path)
    model.to(device)

    model.eval()
    nb_steps, eval_loss, exp_average_loss = 0, 0, None
    score_list = []
    # k = "the book is on the desk. These impressions show , when alive , they had smooth skin , robust limbs with webbed feet , and a ridge of skin on their undersides." tensor(169.6684, device='cuda:0')
    with torch.no_grad():
        for step, s in enumerate(rated_a):  # actually here is a batch with batchsize=1
            # Put model in training mode.
            if not s:
                print('space sentence')
                score_list.append(1e6)
                continue
            #s = s.encode('ascii', 'ignore').decode()
            
            s = enc.encode(s)  # + [50256]  #50256 is the token_id for <|endoftext|>
            batch = torch.tensor([s]).to(device)
            #loss = model(batch, lm_labels=batch)  # everage -logp
            loss = model(batch, labels=batch)['loss']  # everage -logp
            # print(loss*len(s))
            eval_loss += loss.item()
            nb_steps += 1

            score_list.append(loss.item())

    cutoff = np.quantile([-t for t in score_list], 0.05)
    modified_rating = np.array([cutoff if -t < cutoff else -t for t in score_list])
    normed_rating = (modified_rating - cutoff) / np.abs(cutoff)
    return normed_rating



def context_score(questions, answers, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(opt.pretrained_model_path)
    model = GPT2LMHeadModel.from_pretrained(opt.pretrained_model_path)
    model.to(device)

    model.eval()

    score_list = []
    with torch.no_grad():
        for step, (question,answer) in enumerate(zip(questions, answers)):  # actually here is a batch with batchsize=1
            # Put model in training mode.
            if not answer:
                print('space sentence')
                score_list.append(-1e6)

                continue
            
            #question = question.encode('ascii', 'ignore').decode()
            #answer = answer.encode('ascii', 'ignore').decode()

            joint_enc = enc.encode(question+' '+answer)  # + [50256]  #50256 is the token_id for <|endoftext|>
            q = enc.encode(question)
            batch_joint = torch.tensor([joint_enc]).to(device)
            batch_q = torch.tensor([q]).to(device)

            #loss_joint = model(batch_joint, lm_labels=batch_joint) # everage -logp
            #loss_q =  model(batch_q, lm_labels=batch_q)
            loss_joint = model(batch_joint, labels=batch_joint)['loss'] # everage -logp
            loss_q =  model(batch_q, labels=batch_q)['loss']

            p_joint = -loss_joint * (len(joint_enc) -1)
            p_q = -loss_q * (len(q) -1)

            score = p_joint - (p_q)

            score_list.append(score.item())


    cutoff = np.quantile(score_list, 0.05)
    modified_rating = np.array([cutoff if t < cutoff else t for t in score_list])
    normed_rating = (modified_rating - cutoff) / np.abs(cutoff)
    return normed_rating


def diversity_score(cands, ngram):

    score_list = []

    cands = [line.strip() for line in cands]

    for groupid in range(int(len(cands))):
        vocab = {}
        current_sentences = cands[groupid].split('\n')
        for aid in range(len(current_sentences)):
            s = current_sentences[aid]

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

        score_list.append(entropy)

    return np.array(score_list)


def logic_consistency(sentences_pre, sentences):

    score_list = logic_eval(sentences_pre, sentences)

    return np.array(score_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained-model-path', default='gpt2', help='model path of pretrained gpt2 finetuned on dataset')
    parser.add_argument('--metric', default='diversity', help='context | fluency | diversity | logic_consistency')
    parser.add_argument('--file-path', default='test.csv', help='input .csv file')
    parser.add_argument('--output-file-path', default='test_output.csv', help='output .csv file')
    parser.add_argument('--ngram', default=2,
                        help='ngram to be used, for diversity eval')

    opt = parser.parse_args()

    '''
    file follow the following format, no header, answer is generated

    | id 1 | question 1 | answer 1 | 
    | id 2 | question 2 | answer 2 |
    | id 3 | question 3 | answer 3 |
    ...


    '''

    df = pd.read_csv(opt.file_path, header=None)

    if opt.metric == 'context':

        question = df[1].to_list()
        answer = df[2].to_list()
        import ipdb; import sys
        with ipdb.launch_ipdb_on_exception():
            sys.breakpointhook = ipdb.set_trace
            score_list = context_score(question, answer, opt)


    elif opt.metric == 'fluency':

        sentences = df[2].to_list()
        score_list = fluency_score(sentences, opt)

    elif opt.metric == 'diversity':

        ''' answer in each line consists of a group of sentences conditioned on context, separated by newline'''
        sentences = df[2]
        score_list = diversity_score(sentences, opt.ngram)

    elif opt.metric == 'logic_consistency':
        history = df[1].to_list()
        sentences = df[2].to_list()
        score_list = logic_consistency(history, sentences)
    else:
        raise ValueError("Score: context | fluency | diversity | logic_consistency")

    df[3] = pd.Series(score_list)
    df.to_csv(opt.output_file_path)