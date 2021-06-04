from scipy.stats import pearsonr, spearmanr
from openpyxl import load_workbook
import json
import numpy as np
import sys

def correlation(output, score):
    r_spearmanr, p_spearmanr = spearmanr(output, score)
    r_pearsonr, p_pearsonr = pearsonr(output, score)

    spearmanr_res = str(np.round(r_spearmanr, 3)) + ' (' + str(np.round(p_spearmanr, 3)) + ')'
    pearsonr_res = str(np.round(r_pearsonr, 3)) + ' (' + str(np.round(p_pearsonr, 3)) + ')'
    return  [spearmanr_res, pearsonr_res]


if __name__ == "__main__":
    base_dir = sys.argv[1]
    metric_name = sys.argv[2]
    model = sys.argv[3]
    dataset_name = sys.argv[4]
    #HYP_FORMAT = sys.argv[3]
    #CTX_FORMAT = sys.argv[4]
    human_sample_auto_score_fp = open('{}/outputs/{}/grade_data/{}/{}/results.json'.format(base_dir, metric_name, dataset_name, model), 'r')
    spearman_output_file = '{}/outputs/{}/grade_data/{}/{}/spearman_correlation.txt'.format(base_dir, metric_name, dataset_name, model )
    pearson_output_file = '{}/outputs/{}/grade_data/{}/{}/pearson_correlation.txt'.format(base_dir, metric_name, dataset_name, model)
    human_sample_score_file = '{}/data/grade_data/{}/{}/human_score.txt'.format(base_dir, dataset_name, model)


    #output_info = 'Start to compute human_correlation [metric_name: {}, dialog_model: {}, dataset: {}, hyp_format: {}, ctx_format: {}]'.format(
    #    eval_metric_name, model, DATASET_NAME, HYP_FORMAT, CTX_FORMAT)
    #print('-' * len(output_info))
    #print(output_info)
    #print('-' * len(output_info))

    # save score result for correlation
    auto_metric_list = []
    human_sample_score_list = []
    with open(human_sample_score_file, 'r') as fp:
        for line in fp.readlines():
            human_sample_score_list.append(float(line.strip()))
    fp.close()
    auto_score_json = json.load(human_sample_auto_score_fp)


    spearman_res = {}
    pearson_res = {}
    for k,v in auto_score_json.items():
        auto_metric_list = auto_score_json[k]
        #print(len(auto_metric_list), len(human_sample_score_list))
        spearman_p, pearson_p = correlation(auto_metric_list, human_sample_score_list)
        spearman_res[k] = spearman_p
        pearson_res[k] = pearson_p


    print("pearson_res: ", pearson_res, '\n')
    print("spearman_res: ", spearman_res)

    with open(spearman_output_file, 'w') as spearman_f, open(pearson_output_file, 'w') as pearson_f:
        json.dump(spearman_res, spearman_f, indent=2)
        json.dump(pearson_res, pearson_f, indent=2)

    print('------Done------\n')