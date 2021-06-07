#!/usr/bin/env bash

CONTEXT=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt
REFERENCE=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt
RESPONSE=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/ModelPredictions/VHRED/First_VHRED_BeamSearch_5_GeneratedTestResponses.txt_First.txt

ADEM_OUTPUT_FILE=./adem_output.txt \
python entry.py \
    $CONTEXT \
     $REFERENCE \
     $RESPONSE
