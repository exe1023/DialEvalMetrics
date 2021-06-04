import numpy as np
import json

def parse_dialog(dialog):
    '''
    Example:
        ### Dialog: 1
        U: !@#%#@%R
        S: !@##%
        ....
        ### Responses
        S_1: [4,5,6] !@##$@%
        S_2: [1,2,3] !@#$
    '''
    contexts = []
    responses = []
    references = []
    scores = []

    context = []
    response_raw = []
    is_context = True
    for sentence in dialog[1:]:
        if sentence[:3] == '###':
            is_context = False
            continue

        if is_context:
            context.append(sentence[3:].strip())
        else:
            response_raw.append(sentence)
    reference_raw = response_raw[-1]
    reference_split = reference_raw.strip().split()
    reference = ' '.join(reference_split[2:])

    for r_raw in response_raw[:-1]:
        response_split = r_raw.strip().split()
        score = json.loads(response_split[1])
        response = ' '.join(response_split[2:])

        if response.strip() == '':
            response = 'NO RESPONSE' 
        contexts.append(context)
        responses.append(response)
        references.append(reference)
        scores.append(np.mean(score))
    return contexts, responses, references, scores


def load_dstc6_data(base_dir):

    contexts, responses, references, scores = [], [], [], []
    with open(f'{base_dir}/human_rating_scores.txt') as f:
        dialog = []
        for line in f.readlines():
            if line.strip() == '':
                ctx, res, ref, sc = parse_dialog(dialog)
                contexts.extend(ctx)
                responses.extend(res)
                references.extend(ref)
                scores.extend(sc)

                dialog = []
            else:
                dialog.append(line)

    assert len(contexts) == len(responses)
    return {
        'contexts': contexts,
        'responses': responses,
        'references': references,
        'scores': scores
    }

if __name__ == '__main__':
    data = load_dstc6_data('./')
    with open('dstc6_data.json', 'w') as f:
        json.dump(data, f)