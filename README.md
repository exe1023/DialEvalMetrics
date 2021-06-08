# Dialog Evaluation Metrics

## Prerequisties

We use conda to mangage environments for different metrics.

Each directory in `conda_envs` holds an environment specification. Please install all of them before starting the next step.

Take the installation of `conda_envs/eval_base` for example, please run

```
conda create --name eval_base --file conda_envs/eval_base/spec_file.txt
conda env update --name eval_base --file conda_envs/eval_base/environment.yml
```


## Data Preparation

The directory of each qualitiy-annotated data is placed in `data`, with the `data_loader.py` for parsing the data.

Please follow the below instructions to downlaod each dataset, place it into corresponding directory, and run the `data_loader.py` directly to see if you use the correct data.

### DSTC6 Data

Download `human_rating_scores.txt` from https://www.dropbox.com/s/oh1trbos0tjzn7t/dstc6_t2_evaluation.tgz .

### DSTC9 Data

Download and Place the data directory https://github.com/ictnlp/DialoFlow/tree/main/FlowScore/data into `data/dstc9_data`.

### Engage Data

Download https://github.com/PlusLabNLP/PredictiveEngagement/blob/master/data/Eng_Scores_queries_gen_gtruth_replies.csv and rename it to `engage_all.csv`.

### Fed Data

Download http://shikib.com/fed_data.json .

### Grade Data

Download and place each directory in https://github.com/li3cmz/GRADE/tree/main/evaluation/eval_data as `data/grade_data/[convai2|dailydialog|empatheticdialogues]`.

Also download the `human_score.txt` in https://github.com/li3cmz/GRADE/tree/main/evaluation/human_score into the corresponding `data/grade_data/[convai2|dailydialog|empatheticdialogues]`.

### Holistic Data

Download `context_data_release.csv` and `fluency_data_release.csv` from https://github.com/alexzhou907/dialogue_evaluation .

### USR Data

Download TopicalChat and PersonaChat data from http://shikib.com/usr 

## Metric Installation

For baselines, we use the [nlg-eval](https://github.com/Maluuba/nlg-eval).
Please folloow the instruction to install it.

For each dialog metrics, please follow the instructions in README in the corresponding directory.

## Running Notes for Specific metrics


### bert-as-service

PredictiveEngage, BERT-RUBER and PONE requires the running bert-as-service.

If you want to evaluate them, please install and run bert-as-service following the instrucitons [here](https://github.com/hanxiao/bert-as-service).

We also provide a script we used to run bert-as-service `run_bert_as_service.sh`, feel free to use it.

### running USR and FED

We used a web server for running USR and FED in our experiments.

Please run `usr_fed/usr/usr_server.py` and `usr_fed/fed/fed_server.py` to start the server, and modify the path in `usr_fed_metric.py`.


## How to evaluate

1. After you downlaod all datasets, run `gen_data.py` to transform each dataset into the input format for each metrics.

2. Modify the path in `run_eval.sh` as specified in the script, since we need to activate Conda environment when running the script. Run `eval_metrics.sh` to evaluate all quality-anntoated data.

3. Some metrics generate the output in its special format. Therefore, we should run `read_result.py` to read the results of those metrics and transform it into `outputs`

4. The `outputs/METRIC/DATA/results.json` holds the prediction score of each metrics (METRIC) and qualitiy-anntoated data (DATA), while running `data_loader.py` directly in each data directory also generates the corresponding human scores. You can perform any analysis with the data (The jupyter notebook used in our analysis will be released) .

For example, `outputs/grade/dstc9_data/results.json` could be

```

    'GRADE': # the metric name
    [
        0.2568123, # the score of the first sample
        0.1552132, 
        ...
        0.7812346
    ]

```

## Results

All values are statistically significant to p-value < 0.05, unless marked by *.

### USR Data

<table>
    <tr>
        <td></td>
        <td colspan=4>USR-TopicalChat</td>
        <td colspan=4>USR-Pearsonachat</td>
    </tr>
    <tr>
        <td></td>
        <td colspan=2>Turn-Level</td>
        <td colspan=2>System-Level</td>
        <td colspan=2>Turn-Level</td>
        <td colspan=2>System-Level</td>
    </tr>
    <tr>
        <td></td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
    </tr>
    <tr>
        <td>BLEU-4</td>
        <td>0.216</td>
        <td>0.296</td>
        <td>0.874*</td>
        <td>0.900</td>
        <td>0.135</td>
        <td>0.090*</td>
        <td>0.841*</td>
        <td>0.800*</td>
    </tr>
    <tr>
        <td>METEOR</td>
        <td>0.336</td>
        <td>0.391</td>
        <td>0.943</td>
        <td>0.900</td>
        <td>0.253</td>
        <td>0.271</td>
        <td>0.907*</td>
        <td>0.800*</td>
    </tr>
    <tr>
        <td>ROUGE-L</td>
        <td>0.275</td>
        <td>0.287</td>
        <td>0.814*</td>
        <td>0.900</td>
        <td>0.066*</td>
        <td>0.038*</td>
        <td>0.171*</td>
        <td>0.000*</td>
    </tr>
    <tr>
        <td>ADEM</td>
        <td>-0.060*</td>
        <td>-0.061*</td>
        <td>0.202*</td>
        <td>0.700*</td>
        <td>-0.141</td>
        <td>-0.085*</td>
        <td>0.523*</td>
        <td>0.400*</td>
    </tr>
    <tr>
        <td>BERTScore</td>
        <td>0.298</td>
        <td>0.325</td>
        <td>0.854*</td>
        <td>0.900</td>
        <td>0.152</td>
        <td>0.122*</td>
        <td>0.241*</td>
        <td>0.000*</td>
    </tr>
    <tr>
        <td>BLEURT</td>
        <td>0.216</td>
        <td>0.261</td>
        <td>0.630*</td>
        <td>0.900</td>
        <td>0.065*</td>
        <td>0.054*</td>
        <td>-0.125*</td>
        <td>0.000*</td>
    </tr>
    <tr>
        <td>RUBER</td>
        <td>0.247</td>
        <td>0.259</td>
        <td>0.876*</td>
        <td>1.000</td>
        <td>0.131</td>
        <td>0.190</td>
        <td>0.997</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>BERT-RUBER</td>
        <td>0.342</td>
        <td>0.348</td>
        <td>0.992</td>
        <td>0.900</td>
        <td>0.266</td>
        <td>0.248</td>
        <td>0.958</td>
        <td>0.200*</td>
    </tr>
    <tr>
        <td>PONE</td>
        <td>0.271</td>
        <td>0.274</td>
        <td>0.893</td>
        <td>0.500*</td>
        <td>0.373</td>
        <td>0.375</td>
        <td>0.979</td>
        <td>0.800*</td>
    </tr>
    <tr>
        <td>HolisticEval</td>
        <td>-0.147</td>
        <td>-0.123</td>
        <td>-0.919</td>
        <td>-0.200*</td>
        <td>0.087*</td>
        <td>0.113*</td>
        <td>0.051*</td>
        <td>0.000*</td>
    </tr>
    <tr>
        <td>PredictiveEngage</td>
        <td>0.222</td>
        <td>0.310</td>
        <td>0.870*</td>
        <td>0.900</td>
        <td>-0.003*</td>
        <td>0.033*</td>
        <td>0.683*</td>
        <td>0.200*</td>
    </tr>
    <tr>
        <td>MAUDE</td>
        <td>0.044*</td>
        <td>0.083*</td>
        <td>0.317*</td>
        <td>-0.200*</td>
        <td>0.345</td>
        <td>0.298</td>
        <td>0.440*</td>
        <td>0.400*</td>
    </tr>
    <tr>
        <td>GRADE</td>
        <td>0.200</td>
        <td>0.217</td>
        <td>0.553*</td>
        <td>0.100*</td>
        <td>0.358</td>
        <td>0.352</td>
        <td>0.811*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>USR</td>
        <td>0.412</td>
        <td>0.423</td>
        <td>0.967</td>
        <td>0.900</td>
        <td>0.440</td>
        <td>0.418</td>
        <td>0.864*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>FED</td>
        <td>-0.124</td>
        <td>-0.135</td>
        <td>0.730*</td>
        <td>0.100*</td>
        <td>-0.028*</td>
        <td>-0.000*</td>
        <td>0.005*</td>
        <td>0.400*</td>
    </tr>
    <tr>
        <td>Deep AM-FM</td>
        <td>0.285</td>
        <td>0.268</td>
        <td>0.969</td>
        <td>0.700*</td>
        <td>0.228</td>
        <td>0.219</td>
        <td>0.965</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>FlowScore</td>
        <td>0.095*</td>
        <td>0.082*</td>
        <td>-0.150*</td>
        <td>0.400*</td>
        <td>0.118*</td>
        <td>0.079*</td>
        <td>0.678*</td>
        <td>0.800*</td>
    </tr>
</table>

### GRADE data

<table>
    <tr>
        <td></td>
        <td colspan=4>GRADE-ConvAI2</td>
        <td colspan=4>GRADE-DailyDialog</td>
        <td colspan=4>GRADE-EmpatheticDialogue</td>
    </tr>
    <tr>
        <td></td>
        <td colspan=2>Turn-Level</td>
        <td colspan=2>System-Level</td>
        <td colspan=2>Turn-Level</td>
        <td colspan=2>System-Level</td>
        <td colspan=2>Turn-Level</td>
        <td colspan=2>System-Level</td>
    </tr>
    <tr>
        <td></td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
    </tr>
    <tr>
        <td>BLEU-4</td>
        <td>0.003*</td>
        <td>0.128</td>
        <td>0.034*</td>
        <td>0.000*</td>
        <td>0.075*</td>
        <td>0.184</td>
        <td>1.000*</td>
        <td>1.000</td>
        <td>-0.051*</td>
        <td>0.002*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>METEOR</td>
        <td>0.145</td>
        <td>0.181</td>
        <td>0.781*</td>
        <td>0.600*</td>
        <td>0.096*</td>
        <td>0.010*</td>
        <td>-1.000*</td>
        <td>-1.000</td>
        <td>0.118</td>
        <td>0.055*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>ROUGE-L</td>
        <td>0.136</td>
        <td>0.140</td>
        <td>0.209*</td>
        <td>0.000*</td>
        <td>0.154</td>
        <td>0.147</td>
        <td>1.000*</td>
        <td>1.000</td>
        <td>0.029*</td>
        <td>-0.013*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>ADEM</td>
        <td>-0.060*</td>
        <td>-0.057*</td>
        <td>-0.368*</td>
        <td>-0.200*</td>
        <td>0.064*</td>
        <td>0.071*</td>
        <td>1.000*</td>
        <td>1.000</td>
        <td>-0.036*</td>
        <td>-0.028*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>BERTScore</td>
        <td>0.225</td>
        <td>0.224</td>
        <td>0.918*</td>
        <td>0.800*</td>
        <td>0.129</td>
        <td>0.100*</td>
        <td>-1.000*</td>
        <td>-1.000</td>
        <td>0.046*</td>
        <td>0.033*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>BLEURT</td>
        <td>0.125</td>
        <td>0.120</td>
        <td>-0.777*</td>
        <td>-0.400*</td>
        <td>0.176</td>
        <td>0.133</td>
        <td>1.000*</td>
        <td>1.000</td>
        <td>0.087*</td>
        <td>0.051*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>RUBER</td>
        <td>-0.027*</td>
        <td>-0.042*</td>
        <td>-0.458*</td>
        <td>-0.400*</td>
        <td>-0.084*</td>
        <td>-0.094*</td>
        <td>-1.000*</td>
        <td>-1.000</td>
        <td>-0.078*</td>
        <td>-0.039*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>BERT-RUBER</td>
        <td>0.309</td>
        <td>0.314</td>
        <td>0.885*</td>
        <td>1.000</td>
        <td>0.134</td>
        <td>0.128</td>
        <td>-1.000*</td>
        <td>-1.000</td>
        <td>0.163</td>
        <td>0.148</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>PONE</td>
        <td>0.362</td>
        <td>0.373</td>
        <td>0.816*</td>
        <td>0.800*</td>
        <td>0.163</td>
        <td>0.163</td>
        <td>-1.000*</td>
        <td>-1.000</td>
        <td>0.177</td>
        <td>0.161</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>HolisticEval</td>
        <td>-0.030*</td>
        <td>-0.010*</td>
        <td>-0.297*</td>
        <td>-0.400*</td>
        <td>0.025*</td>
        <td>0.020*</td>
        <td>1.000*</td>
        <td>1.000</td>
        <td>0.199</td>
        <td>0.204</td>
        <td>-1.000*</td>
        <td>-1.000</td>
    </tr>
    <tr>
        <td>PredictiveEngage</td>
        <td>0.154</td>
        <td>0.164</td>
        <td>0.601*</td>
        <td>0.600*</td>
        <td>-0.133</td>
        <td>-0.135</td>
        <td>-1.000*</td>
        <td>-1.000</td>
        <td>-0.032*</td>
        <td>-0.078*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>MAUDE</td>
        <td>0.351</td>
        <td>0.304</td>
        <td>0.748*</td>
        <td>0.800*</td>
        <td>-0.036*</td>
        <td>-0.073*</td>
        <td>1.000*</td>
        <td>1.000</td>
        <td>0.007*</td>
        <td>-0.057*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>GRADE</td>
        <td>0.566</td>
        <td>0.571</td>
        <td>0.883*</td>
        <td>0.800*</td>
        <td>0.278</td>
        <td>0.253</td>
        <td>-1.000*</td>
        <td>-1.000</td>
        <td>0.330</td>
        <td>0.297</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>USR</td>
        <td>0.501</td>
        <td>0.500</td>
        <td>0.995</td>
        <td>1.000</td>
        <td>0.057*</td>
        <td>0.057*</td>
        <td>-1.000*</td>
        <td>-1.000</td>
        <td>0.264</td>
        <td>0.255</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>FED</td>
        <td>-0.090</td>
        <td>-0.072*</td>
        <td>-0.254*</td>
        <td>0.000*</td>
        <td>0.080*</td>
        <td>0.064*</td>
        <td>1.000*</td>
        <td>1.000</td>
        <td>-0.014*</td>
        <td>-0.044*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>Deep AM-FM</td>
        <td>0.117</td>
        <td>0.130</td>
        <td>0.774*</td>
        <td>0.400*</td>
        <td>0.026*</td>
        <td>0.022*</td>
        <td>1.000*</td>
        <td>1.000</td>
        <td>0.083*</td>
        <td>0.058*</td>
        <td>1.000*</td>
        <td>1.000</td>
    </tr>
    <tr>
        <td>FlowScore</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>

### DSTC6 data

<table>
    <tr>
        <td></td>
        <td colspan=4>DSTC6</td>
    </tr>
    <tr>
        <td></td>
        <td colspan=2>Turn-Level</td>
        <td colspan=2>System-Level</td>
    </tr>
    <tr>
        <td></td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
    </tr>
    <tr>
        <td>BLEU-4</td>
        <td>0.131</td>
        <td>0.298</td>
        <td>-0.064*</td>
        <td>0.050*</td>
    </tr>
    <tr>
        <td>METEOR</td>
        <td>0.307</td>
        <td>0.323</td>
        <td>0.633</td>
        <td>0.084*</td>
    </tr>
    <tr>
        <td>ROUGE-L</td>
        <td>0.332</td>
        <td>0.326</td>
        <td>0.487</td>
        <td>0.215*</td>
    </tr>
    <tr>
        <td>ADEM</td>
        <td>0.151</td>
        <td>0.118</td>
        <td>0.042*</td>
        <td>0.347*</td>
    </tr>
    <tr>
        <td>BERTScore</td>
        <td>0.369</td>
        <td>0.337</td>
        <td>0.671</td>
        <td>0.265*</td>
    </tr>
    <tr>
        <td>BLEURT</td>
        <td>0.326</td>
        <td>0.294</td>
        <td>0.213*</td>
        <td>0.426*</td>
    </tr>
    <tr>
        <td>RUBER</td>
        <td>0.114</td>
        <td>0.092</td>
        <td>-0.074*</td>
        <td>0.104*</td>
    </tr>
    <tr>
        <td>BERT-RUBER</td>
        <td>0.204</td>
        <td>0.217</td>
        <td>0.825</td>
        <td>0.093*</td>
    </tr>
    <tr>
        <td>PONE</td>
        <td>0.208</td>
        <td>0.200</td>
        <td>0.608</td>
        <td>0.235*</td>
    </tr>
    <tr>
        <td>HolisticEval</td>
        <td>0.001*</td>
        <td>-0.004*</td>
        <td>0.010</td>
        <td>-0.002</td>
    </tr>
    <tr>
        <td>PredictiveEngage</td>
        <td>0.043</td>
        <td>0.004*</td>
        <td>-0.094*</td>
        <td>-0.409*</td>
    </tr>
    <tr>
        <td>MAUDE</td>
        <td>0.195</td>
        <td>0.128</td>
        <td>0.739</td>
        <td>0.217*</td>
    </tr>
    <tr>
        <td>GRADE</td>
        <td>0.119</td>
        <td>0.122</td>
        <td>0.784</td>
        <td>0.611</td>
    </tr>
    <tr>
        <td>USR</td>
        <td>0.184</td>
        <td>0.166</td>
        <td>0.432*</td>
        <td>0.147*</td>
    </tr>
    <tr>
        <td>FED</td>
        <td>-0.106</td>
        <td>-0.083</td>
        <td>0.221*</td>
        <td>0.322*</td>
    </tr>
    <tr>
        <td>Deep AM-FM</td>
        <td>0.326</td>
        <td>0.295</td>
        <td>0.817</td>
        <td>0.674</td>
    </tr>
    <tr>
        <td>FlowScore</td>
        <td>0.064</td>
        <td>0.095</td>
        <td>0.352*</td>
        <td>0.362*</td>
    </tr>
</table>

### PredictiveEngage-DailyDialog

<table>
    <tr>
        <td></td>
        <td colspan=2>PredictiveEngage-DailyDialog</td>
    </tr>
    <tr>
        <td></td>
        <td colspan=2>Turn-Level</td>
    </tr>
    <tr>
        <td></td>
        <td>P</td>
        <td>S</td>
    </tr>
    <tr>
        <td>HolisticEval</td>
        <td>0.368</td>
        <td>0.365</td>
    </tr>
    <tr>
        <td>PredictiveEngage</td>
        <td>0.429</td>
        <td>0.414</td>
    </tr>
    <tr>
        <td>MAUDE</td>
        <td>0.104</td>
        <td>0.060*</td>
    </tr>
    <tr>
        <td>GRADE</td>
        <td>0.600</td>
        <td>0.622</td>
    </tr>
    <tr>
        <td>USR</td>
        <td>0.582</td>
        <td>0.640</td>
    </tr>
    <tr>
        <td>FED</td>
        <td>0.164</td>
        <td>0.159</td>
    </tr>
    <tr>
        <td>FlowScore</td>
        <td>-</td>
        <td>-</td>
    </tr>

</table>


### HolisticEval-DailyDialog

<table>
    <tr>
        <td></td>
        <td colspan=2>HolisticEval-DailyDialog</td>
    </tr>
    <tr>
        <td></td>
        <td colspan=2>Turn-Level</td>
    </tr>
    <tr>
        <td></td>
        <td>P</td>
        <td>S</td>
    </tr>
    <tr>
        <td>HolisticEval</td>
        <td>0.670</td>
        <td>0.764</td> 
    </tr>
    <tr>
        <td>PredictiveEngage</td>
        <td>-0.033*</td>
        <td>0.060*</td>
    <tr>
        <td>MAUDE</td>
        <td>0.275</td>
        <td>0.364</td>
    </tr>
    <tr>
        <td>GRADE</td>
        <td>0.678</td>
        <td>0.697</td>
    </tr>
    <tr>
        <td>USR</td>
        <td>0.589</td>
        <td>0.645</td>
    </tr>
    <tr>
        <td>FED</td>
        <td>0.485</td>
        <td>0.507</td>
    </tr>
    <tr>
        <td>FlowScore</td>
        <td>-</td>
        <td>-</td>
    </tr>

</table>

### FED Data

<table>
    <tr>
        <td></td>
        <td colspan=4>FED</td>
    </tr>
    <tr>
        <td></td>
        <td colspan=2>Turn-Level</td>
        <td colspan=2>Dialog-Level</td>
    </tr>
    <tr>
        <td></td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
    </tr>
    <tr>
        <td>HolisticEval</td>
        <td>0.122</td>
        <td>0.125</td>
        <td>-0.276</td>
        <td>-0.304</td>
    </tr>
    <tr>
        <td>PredictiveEngage</td>
        <td>0.024*</td>
        <td>0.094*</td>
        <td>0.026*</td>
        <td>0.155*</td>
    </tr>
    <tr>
        <td>MAUDE</td>
        <td>0.018*</td>
        <td>-0.094*</td>
        <td>-0.047*</td>
        <td>-0.280</td>
    </tr>
    <tr>
        <td>GRADE</td>
        <td>0.134</td>
        <td>0.118</td>
        <td>-0.034*</td>
        <td>-0.065*</td>
    </tr>
    <tr>
        <td>USR</td>
        <td>0.114</td>
        <td>0.117</td>
        <td>0.093*</td>
        <td>0.062*</td>
    </tr>
    <tr>
        <td>FED</td>
        <td>0.120</td>
        <td>0.095</td>
        <td>0.222</td>
        <td>0.320</td>
    </tr>
    <tr>
        <td>FlowScore</td>
        <td>-0.065*</td>
        <td>-0.055*</td>
        <td>-0.073*</td>
        <td>-0.003*</td>
    </tr>

</table>


<table>
    <tr>
        <td></td>
        <td colspan=4>DSTC9</td>
    </tr>
    <tr>
        <td></td>
        <td colspan=2>Dialog-Level</td>
        <td colspan=2>System-Level</td>
    </tr>
    <tr>
        <td></td>
        <td>P</td>
        <td>S</td>
        <td>P</td>
        <td>S</td>
    </tr>
    <tr>
        <td>HolisticEval</td>
        <td>0.015*</td>
        <td>0.002*</td>
        <td>-0.019*</td>
        <td>-0.100*</td>
    </tr>
    <tr>
        <td>PredictiveEngage</td>
        <td>0.114</td>
        <td>0.115</td>
        <td>0.809</td>
        <td>0.664</td>
    </tr>
    <tr>
        <td>MAUDE</td>
        <td>0.059</td>
        <td>0.042*</td>
        <td>0.224*</td>
        <td>0.045*</td>
    </tr>
    <tr>
        <td>GRADE</td>
        <td>-0.078</td>
        <td>-0.070</td>
        <td>-0.674</td>
        <td>-0.482*</td>
    </tr>
    <tr>
        <td>USR</td>
        <td>0.019*</td>
        <td>0.020*</td>
        <td>0.149*</td>
        <td>0.127*</td>
    </tr>
    <tr>
        <td>FED</td>
        <td>0.128</td>
        <td>0.120</td>
        <td>0.559*</td>
        <td>0.391*</td>
    </tr>
    <tr>
        <td>FlowScore</td>
        <td>0.147</td>
        <td>0.140</td>
        <td>0.907</td>
        <td>0.900</td>
    </tr>

</table>
