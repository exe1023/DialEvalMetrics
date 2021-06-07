import json
import pickle
def main(path, prefix='BERT-RUBER/data/personachat', data_type='train', target='train', output_pkl=True):
    with open(path) as f:
        data = json.load(f)
    
    all_data = []
    for dialog in data[data_type]:
        last_turn = dialog['utterances'][-1]
        context = last_turn['history']
        response = last_turn['candidates'][-1]
        context.append(response)
        all_data.append(context)

    if output_pkl:
        contexts, responses = [], []
        for dialog in all_data:
            for i in range(len(dialog) - 1):
                contexts.append(dialog[i])
                responses.append(dialog[i+1])
    
        with open(f'{prefix}/src-{target}.embed', 'wb') as f:
            pickle.dump(contexts, f)
        with open(f'{prefix}/tgt-{target}.embed', 'wb') as f:
            pickle.dump(responses, f)
   
    else:
        with open(f'{prefix}/src-{target}.txt', 'w') as f_source:
            with open(f'{prefix}/tgt-{target}.txt', 'w') as f_target:
                for dialog in all_data:
                    for i in range(len(dialog) - 1):
                        f_source.write(dialog[i] + '\n')
                        f_target.write(dialog[i+1] + '\n')
    

if __name__ == '__main__':
    main('personachat_self_original.json', data_type='train', target='train')
    main('personachat_self_original.json', data_type='valid', target='dev')
    main('personachat_self_original.json', data_type='valid', target='test')