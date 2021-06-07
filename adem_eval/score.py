import argparse
from utils import create_model_instance, load_file
from agenda.metric_helper import write_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_file')
    parser.add_argument('--ref_file', help='ground truth file')
    parser.add_argument('--model_file')
    parser.add_argument('--score_file')
    args = parser.parse_args()

    context = load_file(args.context_file)
    gt = load_file(args.ref_file)
    model = load_file(args.model_file)
    adem = create_model_instance()
    scores = adem.get_scores(
        contexts=context,
        gt_responses=gt,
        model_responses=model,
    )

    write_score(
        name='ADEM',
        scores=list(scores),
        output=args.score_file,
    )
