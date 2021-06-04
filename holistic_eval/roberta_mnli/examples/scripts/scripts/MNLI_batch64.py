import os
import logging
import itertools
import multiprocessing as mp
import run_experiment


if __name__ == "__main__":
    type = 'MNLI'
    ex_title = 'MNLI_batch64_epoch10'
     # new error alignment graph_gcn_seq_ex58_with_one_head_error
    print_ = True
    opt_dict = dict()
    if type == 'MNLI':
        opt_dict['model_type'] = ['bert']  #
        opt_dict['model_name_or_path'] = ['bert-base-cased']
        opt_dict['task_name'] = ['MNLI']
        opt_dict['do_train'] = ['True']
        opt_dict['do_eval'] = ['True']
        opt_dict['do_lower_case'] = ['True']
        opt_dict['data_dir'] = ['data/MNLI']
        opt_dict['max_seq_length'] = [128]
        opt_dict['per_gpu_eval_batch_size'] = [64]
        opt_dict['per_gpu_train_batch_size'] = [64]
        opt_dict['learning_rate'] = [2e-5]
        opt_dict['num_train_epochs'] = [10.0]
        opt_dict['output_dir'] = ['model/MNLI/original']
        opt_dict['overwrite_output_dir'] = ['True']
        opt_dict['save_steps'] = [50]




    paras_keys = list(opt_dict)
    args = tuple(opt_dict.values())
    paras_list = list(itertools.product(*args))
    examps = len(paras_list)
    ex_ids = list(range(examps))

    #manual_node
    node_list = ['01']
    GPU_list = [[1]]
    config_list = []
    for i, node in enumerate(node_list):
        for GPU in GPU_list[i]:
            config_list.append([node, GPU])
    x=1


    #logger
    logging.basicConfig(filename='scripts/ex.log', level=logging.DEBUG)
    logger = logging.getLogger('')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    processes = []
    for i, paras in enumerate(paras_list):
        ex_id = ex_ids[i]
        node, GPU = config_list[i]
        paras_dict = dict()
        for i in range(len(paras)):
            paras_dict[paras_keys[i]] = paras[i]
        if print_:
            logger.info('title: '+ ex_title+'_'+ str(ex_id)+' GPU:' +node+'_'+str(GPU) +' paras:'+ str(paras_dict))

        p = mp.Process(target=run_experiment.run_training, args=('_'.join([ex_title, str(ex_id)]),type, paras_dict, node,
                                    GPU, logger , print_ ))
        p.start()
        processes.append(p)
