#!/usr/bin/env python
"""
This script computes dialogue embeddings for dialogues found in a text file.
"""

# !/usr/bin/env python

import argparse
import logging
import math
import os
import pickle
import time

import numpy

from .vhred_dialog_encdec import DialogEncoderDecoder
from .vhred_state import prototype_state

logger = logging.getLogger(__name__)


class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time


def parse_args():
    parser = argparse.ArgumentParser("Compute dialogue embeddings from model")

    parser.add_argument("model_prefix",
                        help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("dialogues",
                        help="File of input dialogues (tab separated)")

    parser.add_argument("output",
                        help="Output file")

    parser.add_argument("--verbose",
                        action="store_true", default=False,
                        help="Be verbose")

    return parser.parse_args()


def compute_encodings(joined_contexts, model, model_compute_encoder_state, model_compute_decoder_state, embedding_type):
    # TODO Fix seqlen below
    seqlen = 250
    context = numpy.zeros((seqlen, len(joined_contexts)), dtype='int32')
    context_lengths = numpy.zeros(len(joined_contexts), dtype='int32')

    last_token_position = numpy.zeros(len(joined_contexts), dtype='int32')

    # second_last_utterance_position = numpy.zeros(len(joined_contexts), dtype='int32')

    for idx in range(len(joined_contexts)):
        context_lengths[idx] = len(joined_contexts[idx])
        if context_lengths[idx] < seqlen:
            context[:context_lengths[idx], idx] = joined_contexts[idx]
        else:
            # If context is longer tham max context, truncate it and force the end-of-utterance token at the end
            context[:seqlen, idx] = joined_contexts[idx][0:seqlen]
            context[seqlen - 1, idx] = model.eos_sym
            context_lengths[idx] = seqlen

        eos_indices = list(numpy.where(context[:context_lengths[idx], idx] == model.eos_sym)[0])

        # if len(eos_indices) > 1:
        #    second_last_utterance_position[idx] = eos_indices[-2]
        # else:
        #    second_last_utterance_position[idx] = context_lengths[idx]

        for k in range(seqlen):
            if not context[k, idx] == 0:
                last_token_position[idx] = k

    n_samples = len(joined_contexts)

    # Generate the reversed context
    reversed_context = model.reverse_utterances(context)

    # Compute encoder hidden states
    if embedding_type.upper() == 'CONTEXT':

        encoder_states = model_compute_encoder_state(context, reversed_context, seqlen + 1)
        context_hidden_states = encoder_states[-2]  # hidden state for the "context" encoder, h_s,
        # and last hidden state of the utterance "encoder", h
        latent_hidden_states = encoder_states[-1]  # mean for the stochastic latent variable, z

        output_states = context_hidden_states


    # Compute decoder hidden states
    elif embedding_type.upper() == 'DECODER':
        assert n_samples <= model.bs
        contexts_to_exclude = model.bs - n_samples
        if contexts_to_exclude > 0:
            new_context = numpy.zeros((seqlen, model.bs), dtype='int32')
            new_context_lengths = numpy.zeros(model.bs, dtype='int32')

            new_context[:, 0:n_samples] = context
            new_context_lengths[0:n_samples] = context_lengths

            n_samples = model.bs

        zero_mask = numpy.zeros((seqlen + 1, n_samples), dtype='float32')
        zero_vector = numpy.zeros((n_samples), dtype='float32')
        ones_mask = numpy.zeros((seqlen + 1, n_samples), dtype='float32')

        if hasattr(model, 'latent_gaussian_per_utterance_dim'):
            gaussian_zeros_vector = numpy.zeros((seqlen + 1, n_samples, model.latent_gaussian_per_utterance_dim),
                                                dtype='float32')
        else:
            gaussian_zeros_vector = numpy.zeros((seqlen + 1, n_samples, 2), dtype='float32')

        if hasattr(model, 'latent_piecewise_per_utterance_dim'):
            uniform_zeros_vector = numpy.zeros((seqlen + 1, n_samples, model.latent_piecewise_per_utterance_dim),
                                               dtype='float32')
        else:
            uniform_zeros_vector = numpy.zeros((seqlen + 1, n_samples, 2), dtype='float32')

        decoder_hidden_states = \
        model_compute_decoder_state(context, reversed_context, seqlen + 1, zero_mask, zero_vector,
                                    gaussian_zeros_vector, uniform_zeros_vector, ones_mask)[0]

        if contexts_to_exclude > 0:
            output_states = decoder_hidden_states[:, 0:n_samples, :]
        else:
            output_states = decoder_hidden_states
    else:
        print('FAILURE: embedding_type has to be either CONTEXT or DECODER!')
        assert False

    outputs = numpy.zeros((output_states.shape[1], output_states.shape[2]), dtype='float32')
    for i in range(output_states.shape[1]):
        outputs[i, :] = output_states[last_token_position[i], i, :]

    return outputs


def main(model_prefix, dialogue_file):
    state = prototype_state()

    state_path = model_prefix + "_state.pkl"
    model_path = model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(pickle.load(src))

    logging.basicConfig(level=getattr(logging, state['level']),
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    state['bs'] = 10

    model = DialogEncoderDecoder(state)

    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")

    contexts = [[]]
    lines = open(dialogue_file, "r").readlines()
    if len(lines):
        contexts = [x.strip() for x in lines]

    model_compute_encoder_state = model.build_encoder_function()
    model_compute_decoder_state = model.build_decoder_encoding()
    dialogue_encodings = []

    # Start loop
    joined_contexts = []
    batch_index = 0
    batch_total = int(math.ceil(float(len(contexts)) / float(model.bs)))
    for context_id, context_sentences in enumerate(contexts):
        # Convert contexts into list of ids
        joined_context = []

        if len(context_sentences) == 0:
            joined_context = [model.eos_sym]
        else:
            joined_context = model.words_to_indices(context_sentences.split())

            if joined_context[0] != model.eos_sym:
                joined_context = [model.eos_sym] + joined_context

            if joined_context[-1] != model.eos_sym:
                joined_context += [model.eos_sym]

        joined_contexts.append(joined_context)

        if len(joined_contexts) == model.bs:
            batch_index = batch_index + 1
            logger.debug("[COMPUTE] - Got batch %d / %d" % (batch_index, batch_total))
            encs = compute_encodings(joined_contexts, model, model_compute_encoder_state, model_compute_decoder_state)
            for i in range(len(encs)):
                dialogue_encodings.append(encs[i])

            joined_contexts = []

    if len(joined_contexts) > 0:
        logger.debug("[COMPUTE] - Got batch %d / %d" % (batch_total, batch_total))
        encs = compute_encodings(joined_contexts, model, model_compute_encoder_state, model_compute_decoder_state)
        for i in range(len(encs)):
            dialogue_encodings.append(encs[i])

    return dialogue_encodings


if __name__ == "__main__":
    args = parse_args()

    # Compute encodings
    dialogue_encodings = main(args.model_prefix, args.dialogues)

    # Save encodings to disc
    pickle.dump(dialogue_encodings, open(args.output + '.pkl', 'w'))

    #  THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python compute_dialogue_embeddings.py tests/models/1462302387.69_testmodel tests/data/tvalid_contexts.txt Latent_Variable_Means --verbose --use-second-last-state
