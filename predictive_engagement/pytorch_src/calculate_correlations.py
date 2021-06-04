import csv
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def calculate_pearson_spearman(human_ruber_eval_fname, human_ruber_corr_fname, num_human_per_qr):
    
    '''this function calculates the correlations between human_judgements and engagement scores computed by different models
        Params:
            human_ruber_eval_fname: input file which includes the list of queries and replies with their engagement scores annotated by human or predicted by different models
            human_ruber_corr_fname: output file that includes the correlations between human judgements and different engagement predicting models 
            num_human_per_qr: number of human anntations for each pair of query and reply
    '''

    human_ruber_eval = open(human_ruber_eval_fname, 'r', encoding='utf-8')
    human_ruber_eval_reader = csv.DictReader(human_ruber_eval)
    
    human_ruber_corr = open(human_ruber_corr_fname, 'w')
    headers = ['query', 'response', 'human_score', 'human_score_01_range', 'Eng_Score_MeanPooling', 'Eng_Score_MaxPooling', 'Ruber_Unref_Score', 'CTX_Ruber_Unref_Score', 'CTX_UnrefRuber_avg_EngMeanPooling', 'CTX_UnrefRuber_avg_EngMaxPooling']
    
    human_ruber_corr_writer = csv.DictWriter(human_ruber_corr, headers)
    human_ruber_corr_writer.writeheader()
    
    query_response = set()
    query_response_human_score = {}
    query_response_ruber_unref_score = {}
    query_response_ctx_ruber_unref_score = {}
    query_response_eng_score_mean = {}
    query_response_eng_score_max = {}
    query_response_human_score_range01 = {}
    query_response_Ctxruberunref_avg_engmean = {}
    query_response_Ctxruberunref_avg_engmax = {}
    
    for row in human_ruber_eval_reader:
        query_response.add(row["query"] + "===" + row["response"])
        query_response_human_score[row["query"] + "===" + row["response"]] = 0

    human_ruber_eval = open(human_ruber_eval_fname)
    human_ruber_eval_reader = csv.DictReader(human_ruber_eval)
    for row in human_ruber_eval_reader:
        q_r = row["query"] + "===" + row["response"]
        query_response_human_score[q_r] += float(row["human_score"])
        query_response_ruber_unref_score[q_r] = float(row["Ruber_Unref_Score"]) 
        query_response_ctx_ruber_unref_score[q_r] = float(row["CTX_Unref_Score"]) 
        query_response_eng_score_mean[q_r] = float(row["Eng_Score_MeanPooling"]) 
        query_response_eng_score_max[q_r] = float(row["Eng_Score_MaxPooling"]) 
        query_response_Ctxruberunref_avg_engmean[q_r] = (float(row["CTX_Unref_Score"])+float(row["Eng_Score_MeanPooling"]))/2
        query_response_Ctxruberunref_avg_engmax[q_r] = (float(row["CTX_Unref_Score"])+float(row["Eng_Score_MaxPooling"]))/2

            
    print('number of items in the set of q r is '+str(len(query_response_Ctxruberunref_avg_engmax)))
    
    for q, score in query_response_human_score.items():
        query_response_human_score[q] = score/num_human_per_qr
        
    human_scores_array = list(query_response_human_score.values())
    human_scores_array_01_range = (human_scores_array-np.min(human_scores_array))/(max(human_scores_array)-min(human_scores_array))
    
    print('number of human scores is '+str(len(human_scores_array_01_range)))

    human_scores = []
    ruber_unref_scores = []
    CTXruber_unref_scores = []
    eng_scores_meanpooling = []
    eng_scores_maxpooling = []
    avgctxruber_avg_engmean = []
    avgctxruber_avg_engmax = []
    
    i =0
    for q_r, human_score in query_response_human_score.items():
        score = human_scores_array_01_range[i]
        i+=1
        human_scores.append(score)
        ruber_unref_scores.append(query_response_ruber_unref_score[q_r])
        CTXruber_unref_scores.append(query_response_ctx_ruber_unref_score[q_r])
        eng_scores_meanpooling.append(query_response_eng_score_mean[q_r])
        eng_scores_maxpooling.append(query_response_eng_score_max[q_r])
        avgctxruber_avg_engmean.append(query_response_Ctxruberunref_avg_engmean[q_r])
        avgctxruber_avg_engmax.append(query_response_Ctxruberunref_avg_engmax[q_r])
        row_write ={}
        row_write['query'] = q_r.split('===')[0]
        row_write['response'] = q_r.split('===')[1]
        row_write['human_score'] = round(query_response_human_score[q_r],4)           
        row_write['human_score_01_range'] = round(score,4)   
        row_write['Ruber_Unref_Score'] = query_response_ruber_unref_score[q_r]
        row_write['CTX_Ruber_Unref_Score'] = query_response_ctx_ruber_unref_score[q_r]
        row_write['Eng_Score_MeanPooling'] = query_response_eng_score_mean[q_r]
        row_write['Eng_Score_MaxPooling'] = query_response_eng_score_max[q_r]
        row_write['CTX_UnrefRuber_avg_EngMeanPooling'] = query_response_Ctxruberunref_avg_engmean[q_r]
        row_write['CTX_UnrefRuber_avg_EngMaxPooling'] = query_response_Ctxruberunref_avg_engmax[q_r]
        human_ruber_corr_writer.writerow(row_write)
    
    

    row_write ={}
    pearson_ref,  pp_value  = pearsonr(human_scores, ruber_unref_scores)
    spearman_corr, p_value = spearmanr(human_scores, ruber_unref_scores)
    row_write['query'] = 'pearson_correlation_unref_score = '+ str(pearson_ref)  + ' p<'+str(pp_value)
    row_write['response'] = 'spearman_correlation_unref_score = '+ str(spearman_corr) + ' p<'+str(p_value)
    human_ruber_corr_writer.writerow(row_write)  


    row_write ={}
    pearson_ref,  pp_value  = pearsonr(human_scores, CTXruber_unref_scores)
    spearman_corr, p_value = spearmanr(human_scores, CTXruber_unref_scores)
    row_write['query'] = 'pearson_correlation_CTXunref_score = '+ str(pearson_ref)  + ' p<'+str(pp_value)
    row_write['response'] = 'spearman_correlation_CTXunref_score = '+ str(spearman_corr) + ' p<'+str(p_value)
    human_ruber_corr_writer.writerow(row_write)  


    row_write ={}
    pearson_ref,  pp_value  = pearsonr(human_scores, eng_scores_meanpooling)
    spearman_corr, p_value = spearmanr(human_scores, eng_scores_meanpooling)
    row_write['query'] = 'pearson_correlation_engmean_score = '+ str(pearson_ref)  + ' p<'+str(pp_value)
    row_write['response'] = 'spearman_correlation_engmean_score = '+ str(spearman_corr) + ' p<'+str(p_value)
    human_ruber_corr_writer.writerow(row_write)  


    row_write ={}
    pearson_ref,  pp_value  = pearsonr(human_scores, eng_scores_maxpooling)
    spearman_corr, p_value = spearmanr(human_scores, eng_scores_maxpooling)
    row_write['query'] = 'pearson_correlation_engmax_score = '+ str(pearson_ref)  + ' p<'+str(pp_value)
    row_write['response'] = 'spearman_correlation_engmax_score = '+ str(spearman_corr) + ' p<'+str(p_value)
    human_ruber_corr_writer.writerow(row_write)  


    row_write ={}
    pearson_avg, pp_value = pearsonr(human_scores, avgctxruber_avg_engmean)
    spearman_corr, p_value = spearmanr(human_scores, avgctxruber_avg_engmean)
    row_write['query'] = 'pearson_correlation_score_avgctxRuber_avg_engmean = '+ str(pearson_avg) + ' p<'+str(pp_value)
    row_write['response'] = 'spearman_correlation_score_avgctxRuber_avg_engmean = '+ str(spearman_corr) + ' p<'+str(p_value)
    human_ruber_corr_writer.writerow(row_write) 

    row_write ={}
    pearson_avg, pp_value = pearsonr(human_scores, avgctxruber_avg_engmax)
    spearman_corr, p_value = spearmanr(human_scores, avgctxruber_avg_engmax)
    row_write['query'] = 'pearson_correlation_score_avgctxRuber_avg_engmax = '+ str(pearson_avg) + ' p<'+str(pp_value)
    row_write['response'] = 'spearman_correlation_score_avgctxRuber_avg_engmax = '+ str(spearman_corr) + ' p<'+str(p_value)
    human_ruber_corr_writer.writerow(row_write) 
    
    

    row_write['query'] = 'cosine_sim_unref = '+ str(cosine_similarity([human_scores], [ruber_unref_scores]))
    row_write['response'] = 'cosine_sim_unref = '+ str(cosine_similarity([human_scores], [CTXruber_unref_scores]))
    human_ruber_corr_writer.writerow(row_write)
    row_write['query'] = 'cosine_sim_engmean = '+ str(cosine_similarity([human_scores], [avgctxruber_avg_engmean]))
    row_write['response'] = 'cosine_sim_engmax = '+ str(cosine_similarity([human_scores], [eng_scores_maxpooling]))
    human_ruber_corr_writer.writerow(row_write)    
    row_write['query'] = 'cosine_sim_avg_CtxRuber_engmean_rel = '+ str(cosine_similarity([human_scores], [avgctxruber_avg_engmean]))
    row_write['response'] = 'cosine_sim_avg_CtxRuber_engmax_rel = '+ str(cosine_similarity([human_scores], [avgctxruber_avg_engmax]))
    human_ruber_corr_writer.writerow(row_write)    
      
  
 
if __name__ == "__main__":
    human_ruber_eval_method1_fname = "./../data/Eng_Scores_queries_gen_gtruth_replies.csv"
    human_ruber_method1_pearson = "./../data/Eng_Scores_queries_gen_gtruth_replies_corr.csv"
    calculate_pearson_spearman(human_ruber_eval_method1_fname, human_ruber_method1_pearson, 3)

