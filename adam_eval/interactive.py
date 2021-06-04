from models import *
from preprocess import Preprocessor

saved_model = './weights/adem_model.pkl'

if __name__ == '__main__':
    pp = Preprocessor()
    adem = ADEM(pp, None, saved_model)

    contexts = ['</s> <first_speaker> hello . how are you today ? </s>',
                '</s> <first_speaker> i love starbucks coffee </s>']
    true = ['</s> <second_speaker> i am fine . thanks </s>',
            '</s> <second_speaker> i like their latte </s>']
    model = ['</s> <second_speaker> fantastic ! how are you ? </s>',
             '</s> <second_speaker> me too ! better than timmies </s>']

    print('Model Loaded!')
    print(adem.get_scores(contexts, true, model))
