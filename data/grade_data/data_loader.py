from pathlib import Path

def load_grade_data(base_dir, dataset, model_name):
    base_dir = Path(f'{base_dir}/{dataset}/{model_name}')
    
    contexts = []
    with (base_dir / 'human_ctx.txt').open() as f:
        for line in f.readlines():
            context = line.strip().split('|||')
            contexts.append(context)
    
    responses = []
    with (base_dir / 'human_hyp.txt').open() as f:
        for line in f.readlines():
            response = line.strip()
            responses.append(response)
    
    references = []
    with (base_dir / 'human_ref.txt').open() as f:
        for line in f.readlines():
            reference = line.strip()
            references.append(reference)
     
    return {
        'contexts': contexts,
        'responses': responses,
        'references': references
    }


if __name__ == '__main__':
    load_grade_data('convai2', 'bert_ranker')