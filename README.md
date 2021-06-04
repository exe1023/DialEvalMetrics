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

### Holistic Data

Download `context_data_release.csv` and `fluency_data_release.csv` from https://github.com/alexzhou907/dialogue_evaluation .

### USR Data

Download TopicalChat and PersonaChat data from http://shikib.com/usr 

## Download Pretrained Models


## How to evaluate

1. After you downlaod all datasets, run `gen_data.py` to transform each dataset into the input format for each metrics.

2. Modify the path in `run_eval.sh` as specified in the script, since we need to activate Conda environment when running the script. Run `eval_metrics.sh` to evaluate all quality-anntoated data.

3. Some metrics generate the output in its special format. Therefore, we should run `read_result.py` to read the results of those metrics and transform it into `outputs`

4. The `outputs/METRIC/DATA/results.json` holds the prediction score of each metrics (METRIC) and qualitiy-anntoated data (DATA), while running `data_loader.py` directly in each data directory also generates the corresponding human scores. Run any analysis with the data (The jupyter notebook used in our analysis will be released) .

