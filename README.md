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
{
    'GRADE': # the metric name
    [
        0.2568123, # the score of the first sample
        0.1552132, 
        ...
        0.7812346
    ]

```

## Results

### USR Data

|                  <td colspan=4> USR-TopicalChat  <td colspan=4> USR-Pearsonachat |
|------------------|-------------------------------------|--------------------------------------|
|                  <td colspan=2> Turn-Level      <td colspan=2> System-Level     | <td colspan=2>Turn-Level <td colspan=2> System-Level |
|                  | P                                   | S                                    | P                              | S                                | P              | S              | P              | S              |
| BLEU-4           | 0.216                               | 0.296                                | 0.874*                         | 0.900                            | 0.135          | 0.090*         | 0.841*         | 0.800*         |
| METEOR           | 0.336                               | 0.391                                | 0.943                          | 0.900                            | 0.253          | 0.271          | 0.907*         | 0.800*         |
| ROUGE-L          | 0.275                               | 0.287                                | 0.814*                         | 0.900                            | 0.066*         | 0.038*         | 0.171*         | 0.000*         |
| ADEM             | -0.060*                             | -0.061*                              | 0.202*                         | 0.700*                           | -0.141         | -0.085*        | 0.523*         | 0.400*         |
| BERTScore        | 0.298                               | 0.325                                | 0.854*                         | 0.900                            | 0.152          | 0.122*         | 0.241*         | 0.000*         |
| BLEURT           | 0.216                               | 0.261                                | 0.630*                         | 0.900                            | 0.065*         | 0.054*         | -0.125*        | 0.000*         |
| RUBER            | 0.247                               | 0.259                                | 0.876*                         | 1.000                   | 0.131          | 0.190          | 0.997 | 1.000 |
| BERT-RUBER       | 0.342                               | 0.348                                | 0.992                 | 0.900                            | 0.266          | 0.248          | 0.958          | 0.200*         |
| PONE             | 0.271                               | 0.274                                | 0.893                          | 0.500*                           | 0.373          | 0.375          | 0.979          | 0.800*         |
| HolisticEval     | -0.147                              | -0.123                               | -0.919                         | -0.200*                          | 0.087*         | 0.113*         | 0.051*         | 0.000*         |
| PredictiveEngage | 0.222                               | 0.310                                | 0.870*                         | 0.900                            | -0.003*        | 0.033*         | 0.683*         | 0.200*         |
| MAUDE            | 0.044*                              | 0.083*                               | 0.317*                         | -0.200*                          | 0.345          | 0.298          | 0.440*         | 0.400*         |
| GRADE            | 0.200                               | 0.217                                | 0.553*                         | 0.100*                           | 0.358          | 0.352          | 0.811*         | 1.000 |
| USR              | 0.412                      | 0.423                       | 0.967                          | 0.900                            | 0.440 | 0.418 | 0.864*         | 1.000 |
| FED              | -0.124                              | -0.135                               | 0.730*                         | 0.100*                           | -0.028*        | -0.000*        | 0.005*         | 0.400*         |
| Deep AM-FM       | 0.285                               | 0.268                                | 0.969                          | 0.700*                           | 0.228          | 0.219          | 0.965          | 1.000 |
| FlowScore        | 0.095*                              | 0.082*                               | -0.150*                        | 0.400*                           | 0.118*         | 0.079*         | 0.678*         | 0.800*         |
| %Mean            | 0.375                               | 0.379                                | 0.936                          | 0.900                            | 0.484          | 0.498          | 0.900*         | 1.000          |
