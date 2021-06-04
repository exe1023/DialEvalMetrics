import json


def make_wordnet_para_intf(inf, dev_matched_tsv, samples_json):
    mode = 'baseline'
    with open(inf, encoding='utf8') as f:  # 'self_logic/MNLI_SCORE/data/{}/to_label_{}.txt'.format(mode,mode)
        lines = f.readlines()
    target_lines = ['index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	label2	label3	label4	label5	gold_label\n']
    id = 0
    sample= dict()
    samples = []
    printed_lines =[]
    sample_line = []
    used_id = 1
    to_para = []
    for i in range(len(lines)):
        if lines[i].strip() == '':
            continue
        if lines[i][0] is not '\t':
            if len(sample)>0 and len(sample['history']) >= 4:
                to_para.append(sample['history'][-4])
                sample['orig_history'] = sample['history'][:-2]
                if len(sample['history'])%2==0:
                    sample['history'] = [sample['history'][a] for a in range(len(sample['history'])) if a % 2 == 1]
                else:
                    sample['history'] = [sample['history'][a] for a in range(len(sample['history'])) if a % 2 == 0]
                sample['pairs'] = []
                for j in range(len(sample['history'])-1):
                    sample['pairs'].append([sample['history'][j], sample['history'][-1]])
                    t_line = str(id) + '\t' + ''.join(['x', '\t'] * 7) + sample['history'][j] + '\t' + sample['history'][-1] + ''.join(
                        ['\t', 'neutral'] * 6)
                    target_lines.append(t_line + '\n')
                if True:
                    samples.append(sample)
                    printed_lines.extend(sample_line)
                    used_id +=1
                id +=1
            sample_line = []
            sample_line.append('sample ' + str(used_id) + ':\n')
            sample = dict()
            sample['history'] = []
        elif lines[i].strip().startswith('YOUR SCORE IS:'):
            continue
        else:
            sample_line.append(lines[i])
            sample['history'].append(lines[i].strip().replace('history: ','').replace('pred: ', ''))
        # else:
        #     sample['history'].extend(lines[i].strip().strip('history: ').strip('pred: ').split('</d>'))
    x=1
    if len(sample) > 0 and len(sample['history']) >= 4:
        to_para.append(sample['history'][-4])
        sample['orig_history'] = sample['history'][:-2]
        if len(sample['history']) % 2 == 0:
            sample['history'] = [sample['history'][a] for a in range(len(sample['history'])) if a % 2 == 1]
        else:
            sample['history'] = [sample['history'][a] for a in range(len(sample['history'])) if a % 2 == 0]
        sample['pairs'] = []
        for j in range(len(sample['history']) - 1):
            sample['pairs'].append([sample['history'][j], sample['history'][-1]])
            t_line = str(id) + '\t' + ''.join(['x', '\t'] * 7) + sample['history'][j] + '\t' + sample['history'][
                -1] + ''.join(
                ['\t', 'neutral'] * 6)
            target_lines.append(t_line + '\n')
        if True:
            samples.append(sample)
            printed_lines.extend(sample_line)
        id += 1

    with open(dev_matched_tsv, 'w', encoding='utf8') as f:
        f.writelines(target_lines)
    with open(samples_json, 'w', encoding='utf8') as f:
        json.dump(samples, f)