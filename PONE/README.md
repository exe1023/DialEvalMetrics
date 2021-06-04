# EPN-RUBER
## Introduction
Implementation of PONE: A Feasible automatic evaluation of Open-Domain Dialog Systems with Enhancing positive and negative samples.

Paper for ACL-2020

1. Based on the BERT-RUBER: 
    
    Better Automatic Evaluation of Open-Domain Dialogue Systems with Contextualized Embeddings

2. Enhancing positive samples

    * OpenNMT-py
    * EDA Data Augmentation (lack of semantic diversity)
    * refer to D3Q 
    * refer to GAN


3. Enhancing negative samples from dataset by using BERT embedding
    
4. More Interpretability
    
    * Fluency 
    * Coherence
    * ......
    
## Performance

Details can be found in `result.ipynb`
    
## Note
* big prechoice size is good? (500 -> 5000)
* test
    1. 5000 / 0.01, this one is bad
    2. 5000 / 0.1, this one is good
* large dataset without weight: performance is bad because of a large number of noise data.
* large dataset with weight (best performance of prechoice and weight ratio): bad too
* Positive method is good, but positive method and negative method is bad ?
    * negative method is not good and degrade the performance ?
* negative method bad
    * so low
    * not so good, why, what the hell.
* the better performance will come up with better test acc