import os
os.environ["THEANO_FLAGS"] = "floatX=float32"
from adam_eval.models import *
from adam_eval.preprocess import Preprocessor


def process_text(texts, speaker):
    adam_text = []
    for text in texts:
        adam_text.append(f'</s> {speaker} {text} </s>')
    return adam_text

def eval_adam(base_dir, contexts, refs, hyps):
    contexts = process_text(contexts, '<first_speaker>')
    refs = process_text(refs, '<second_speaker>')
    hyps = process_text(hyps, '<second_speaker>')

    saved_model = f'{base_dir}/adam_eval/weights/adem_model.pkl'
    pp = Preprocessor()
    adem = ADEM(pp, None, saved_model)
    
    return adem.get_scores(contexts, refs, hyps)

if __name__ == '__main__':

    contexts = ['hello . how are you today ?',
                'i love starbucks coffee']
    true = ['i am fine . thanks',
            'i like their latte']
    model = ['fantastic ! how are you ?',
             'me too ! better than timmies']
    print(eval_adam(contexts, true, model))
 