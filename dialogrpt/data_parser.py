from pathlib import Path
import json

def gen_dialogrpt_data(data, target_dir):
    from transformers import GPT2Tokenizer 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{target_dir}/data.tsv', 'w') as f:
        for context, hyp in zip(data['contexts'], data['responses']):
            context = ' '.join(context).replace('\t', ' ')
            hyp = hyp.replace('\t', ' ')
            
            ctx_token = tokenizer.tokenize(context)
            hyp_token = tokenizer.tokenize(hyp)
            exceed = len(ctx_token) + len(hyp_token) - 1022
            
            if exceed > 0:
                context = ' '.join(context.split()[exceed:])
            
            write_str = f'{context}\t{hyp}\n'
            f.write(write_str)
    
    
def read_dialogrpt_result(data_dir):
    scores = []
    with open(f'{data_dir}/data.tsv.ranked.jsonl') as f:
        for line in f.readlines():
            data = json.loads(line)
            final = data['hyps'][0][0]
            scores.append(final)

    return scores