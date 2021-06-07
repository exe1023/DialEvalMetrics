import logging
import os
import argparse
from pathlib import Path

ADEM_MODEL = os.path.join(os.path.dirname(__file__), 'weights', 'adem_model.pkl')

logger = logging.getLogger(__name__)


def create_model_instance():
    from models import ADEM
    from preprocess import Preprocessor

    logger.info('loading model from %s', ADEM_MODEL)
    model = ADEM(Preprocessor(), None, ADEM_MODEL)
    logger.info('model loaded. config: %r', model.config)
    return model


def load_file(filename):
    logger.info('loading %s', filename)
    with open(filename) as f:
        return f.readlines()


def run():
    ARGS = ('contexts', 'references', 'responses')
    parser = argparse.ArgumentParser()
    for arg in ARGS:
        parser.add_argument(arg)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    payload = {name: load_file(getattr(args, name)) for name in ARGS}

    model = create_model_instance()

    scores = model.get_scores(
        contexts=payload['contexts'],
        gt_responses=payload['references'],
        model_responses=payload['responses'],
    )
    output_file = Path('./adem_output.txt')
    logger.info('Saving scores to {}'.format(output_file.absolute()))
    output_file.write_text('\n'.join(map(str, scores)))
