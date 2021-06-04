This repository has the code and parameters used for the ADEM model in:

**Towards an Automatic Turing Test: Learning to Evaluate Dialogue Responses**  
Ryan Lowe, Michael Noseworthy, Iulian V. Serban, Nicolas Angelard-Gontier, Yoshua Bengio, and Joelle Pineau

Due to the ethics policy for this project, we cannot release the collected human data at this time.
However, we do provide the weights/parameters for a trained model and the code to train ADEM with new data.

ADEM uses the VHRED model. A modified version of the code is included in this repo. The original repo and paper can be found at:  
https://github.com/julianser/hed-dlg-truncated  
https://arxiv.org/abs/1605.06069

You will need to download the weights for the pretrained VHRED model before running the code. Once downloaded from the following link, place all the files in the `./vhred` folder.  
https://drive.google.com/file/d/0B-nb1w_dNuMLY0Fad3N1YU9ZOU0/view?usp=sharing

An example of running ADEM can be found in `interactive.py`:  
`THEANO_FLAGS='device=gpu0,floatX=float32' python interactive.py`

Recommended docker image:

    cgsdfc/adem-1-master:latest
    