from flask import Flask
from flask import json
from flask import make_response
from flask import request

from utils import create_model_instance

app = Flask(__name__)
Adem = create_model_instance()

CONTEXT = 'context'
REFERENCE = 'reference'
RESPONSE = 'response'
SCORE = 'score'

ADEM_INTERNAL_EC = 400


def _make_error(exception):
    return make_response(str(exception), ADEM_INTERNAL_EC)


def _extract_data(dict):
    return dict[CONTEXT], dict[REFERENCE], dict[RESPONSE]


def _load_file(filename):
    with open(filename) as f:
        return f.readlines()


def _make_score(scores):
    return json.dumps({
        SCORE: scores
    })


@app.route('/adem/v1/config')
def get_status():
    return json.dumps(Adem.config)


@app.route('/adem/v1/score/utterance')
def score_utterance():
    """
    Get score using url params. Valid params are:

        ?context=
        ?response=
        ?reference=

    Each denotes an utterance.

    :return: a response whose body is a solo string containing the utterance level score.
    """
    data = [[item] for item in _extract_data(request.args)]
    scores = Adem.get_scores(*data)
    return str(scores[0])


@app.route('/adem/v1/score/corpus')
def score_corpus():
    """
    Get score using request body as a json. Request format is:

        {
            "context": ['c1', 'c2'],
            "reference": ['r1', 'r2'],
            "response": ['r1', 'r2']
        }

    :return: a response whose body is a json. Response format is:

        {
            "score": [s1, s2]
        }
    """
    data = _extract_data(request.json)
    scores = Adem.get_scores(*data)
    return _make_score(scores)


@app.route('/adem/v1/score/files')
def score_files():
    """
    Get score using filename in the request body as a json. Request format is:

         {
            "context": 'path/to/context',
            "reference": 'path/to/reference' ,
            "response": 'path/to/response'
        }


    :return: a response whose body is a json. Response format is:

        {
            "score": [s1, s2]
        }
    """
    filenames = _extract_data(request.json)
    data = [_load_file(filename) for filename in filenames]
    scores = Adem.get_scores(*data)
    return _make_score(scores)
