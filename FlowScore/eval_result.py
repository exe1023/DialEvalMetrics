import json
import os
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np

data = json.load(open("results/dstc9.json"))

print(pearsonr(data['flow_scores'], data['scores']))

