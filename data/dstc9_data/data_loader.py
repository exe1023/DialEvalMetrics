from pathlib import Path
import json

# for checking max length
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def load_dstc9_data(base_dir, max_words=None):

    contexts, responses, references, scores, models = [], [], [], [], []
    path = Path(base_dir + '/data')
    for f in path.iterdir():
        human_dialogues = []
        temp_score = []
        human_scores = []
        model_name = f.name
        print(model_name)

        data_human = json.load(f.open())
        for i in range(len(data_human)):
            if len(human_dialogues) > 0 and data_human[i]["context"] == human_dialogues[-1]:
                temp_score.append(data_human[i]["human (overall)"])
            else:
                if len(temp_score) > 0:
                    scores.append(sum(temp_score) / len(temp_score))
                    
                human_dialogues.append(data_human[i]["context"])
                temp_score = [data_human[i]["human (overall)"]]
                
                a = [k.replace("User: ", "").replace("System: ", "") for k in
                    data_human[i]["context"].split("\n")[:-1]]
                a = [x for x in a if len(x.strip()) > 0]
                
                # check the maximum length of the context
                if max_words is not None:
                    words_num = len(tokenizer.tokenize(a[-1]))
                    
                    dialog_context = a[:-1]
                    context = []
                    for text in dialog_context[::-1]:
                        words_num += len(tokenizer.tokenize(text))
                        if words_num < max_words:
                            context.append(text)
                        else:
                            break
                    
                    if len(context) == 0:
                        truncated = dialog_context[-1].split()
                        context.append(' '.join(truncated[-256:]))
                    
                    context = context[::-1]
                else:
                    context = a[:-1]

                response = a[-1]
                if len(contexts) == 1679: # poisoned example
                    response = response[-512:]
                
                contexts.append(context)
                responses.append(response)
                references.append('NO REF')
                models.append(model_name)

        scores.append(sum(temp_score) / len(temp_score))


    assert len(contexts) == len(responses) and len(contexts) == len(scores)
    assert len(contexts) == len(models)
    return {
        'contexts': contexts,
        'responses': responses,
        'references': references,
        'scores': scores,
        'models': models
    }


if __name__ == '__main__':
    data = load_dstc9_data('./')
    for key, item in data.items():
        print(len(item))
    with open('dstc9_data.json', 'w') as f:
        json.dump(data, f)