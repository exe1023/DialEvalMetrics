import numpy as np
import pickle

scores = {'dailydialog': {'HRED': [13.41], 'HRAN': [15.61], 'WSeq': [13.10], 'WSeq_RA': [14.93], 'DSHRED': [14.43], 'DSHRED_RA': [16.48], 'ReCoSa': [15.67], 'ReCoSa_RA': [14.45]},
          'empchat': {'HRED': [13.17], 'HRAN': [13.05], 'WSeq': [12.99], 'WSeq_RA': [13.64], 'DSHRED': [13.16], 'DSHRED_RA': [13.54], 'ReCoSa': [12.72], 'ReCoSa_RA': [12.66]},
          'personachat': {'HRED': [15.82], 'HRAN': [16.99], 'WSeq': [15.96], 'WSeq_RA': [16.52], 'DSHRED': [15.96], 'DSHRED_RA': [16.68], 'ReCoSa': [15.65], 'ReCoSa_RA': [15.34]}}

with open('bert-ptest.txt') as f:
    for line in f.readlines():
        line = line.strip().split()
        if len(line) != 4:
            pass
        else:
            dataset, model, _, score = line
            model = model.split('-')[0]
            scores[dataset][model].append(100 * float(score))

with open('bertscore.pkl', 'wb') as f:
    pickle.dump(scores, f)

for dataset in scores.keys():
    for model in scores[dataset].keys():
        if len(scores[dataset][model]) == 11:
            s = np.array(scores[dataset][model])
            s = (s-s[0])[1:]
            assert len(s) == 10
            print(f'{dataset}, {model}, perturbation test: {round(np.mean(s), 4)}')
