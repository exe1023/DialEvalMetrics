from tqdm import tqdm
import sys
import os
import shutil

model = sys.argv[1]
EVAL_DATASET_NAME = sys.argv[2]
HYP_FORMAT = sys.argv[3]
CTX_FORMAT = sys.argv[4]


output_info = 'Prepare keyword and text data for evaluate [dialog_model: {}, dataset: {}, hyp_format: {}, ctx_format: {}]'.format(
        model, EVAL_DATASET_NAME, HYP_FORMAT, CTX_FORMAT)
print('-' * len(output_info))
print(output_info)
print('-' * len(output_info))


def maybe_create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for mode in ['test']:
        folder = os.path.join(dir_name, mode)
        if not os.path.exists(folder):
            os.mkdir(folder)
        for pair in ['pair-1']:
            sub_folder = os.path.join(folder, pair)
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)

root_dir = '../data/{}'.format(EVAL_DATASET_NAME)
maybe_create_dir(root_dir)
ori_output_file_path = '../data/{}/test/pair-1/'.format(EVAL_DATASET_NAME)

ctx = './eval_data/{}/{}/human_{}.keyword'.format(EVAL_DATASET_NAME, model, CTX_FORMAT)
hyp = './eval_data/{}/{}/human_{}.keyword'.format(EVAL_DATASET_NAME, model, HYP_FORMAT)
output_ori = open('{}/original_dialog.keyword'.format(ori_output_file_path),'w')

output_ori_merge_f = open('{}/original_dialog_merge.keyword'.format(ori_output_file_path),'w')
output_ori_merge_ctx_f = open('{}/original_dialog_merge.ctx_keyword'.format(ori_output_file_path),'w')
output_ori_merge_rep_f = open('{}/original_dialog_merge.rep_keyword'.format(ori_output_file_path),'w')

with open(ctx, 'r') as ctx_f, open(hyp, 'r') as hyp_f:
    for c,h in tqdm(zip(ctx_f.readlines(), hyp_f.readlines())):
        original_keyword = c.strip() + '|||' + h.strip()

        output_ori.writelines([original_keyword+'\n'])

        ctx_key_merge = ' '.join(c.strip().split("|||"))
        rep_key_merge = h.strip()
        original_keyword_merge = ctx_key_merge + ' ' + rep_key_merge
        original_keyword_merge_ctx = ctx_key_merge
        original_keyword_merge_rep = rep_key_merge

        output_ori_merge_f.writelines([original_keyword_merge + '\n'])
        output_ori_merge_ctx_f.writelines([original_keyword_merge_ctx + '\n'])
        output_ori_merge_rep_f.writelines([original_keyword_merge_rep + '\n'])

output_ori.close()
output_ori_merge_f.close()
output_ori_merge_ctx_f.close()
output_ori_merge_rep_f.close()


ctx = './eval_data/{}/{}/human_{}.txt'.format(EVAL_DATASET_NAME, model, CTX_FORMAT)
hyp = './eval_data/{}/{}/human_{}.txt'.format(EVAL_DATASET_NAME, model, HYP_FORMAT)
output_ori = open('{}/original_dialog.text'.format(ori_output_file_path),'w')
with open(ctx, 'r') as ctx_f, open(hyp, 'r') as hyp_f:
    for c,h in tqdm(zip(ctx_f.readlines(), hyp_f.readlines())):
        original_text = c.strip() + '|||' + h.strip()

        output_ori.writelines([original_text+'\n'])
output_ori.close()

print('Done.\n')