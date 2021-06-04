"""
This file will clean up the data collected from AMT.

"""
import pickle
import re


class AMT_DataLoader(object):

    def __init__(self, preprocessor, config):
        self.amt_folder = config['raw_data_location']
        self.preprocessor = preprocessor

    def _fix_modelmaps(self):
        with open('%s/models.pkl' % self.amt_folder, 'rb') as handle:
            model_map = pickle.load(handle)
        with open('%s/models_new.pkl' % self.amt_folder, 'rb') as handle:
            model_map2 = pickle.load(handle)
        # For ids less than 988, we will use the data collected from the first collection round.
        for key in list(model_map2.keys()):
            if key <= 988:
                del model_map2[key]
        model_map = dict(model_map, **model_map2)
        return model_map

    def _combine_contextids(self, c_id1, c_id2):
        # Combines 2 lists of context ids
        for c_id in c_id2:
            if c_id not in c_id1:
                c_id1.append(c_id)
        c_id1.sort()
        return c_id1

    def _filter_modelmap(self, model_map, valid_contextids):
        for key in list(model_map.keys()):
            if key not in valid_contextids:
                del model_map[key]
        return model_map

    def _get_twitter_data(self, clean_data_file, context_file, gt_file, model_map):
        '''
        Loads Twitter data from dictionaries.
        '''
        with open(clean_data_file, 'rb') as handle:
            clean_data = pickle.load(handle)
        with open(context_file, 'rb') as handle:
            contexts = pickle.load(handle)
        with open(gt_file, 'r') as handle:
            gt_unordered = handle.readlines()

        # Score_dic will be indexed by context_ids.
        score_dic = {}
        # First iterate over HIT ids.
        for user in clean_data:
            # Then iterate over context that user completed.
            for dic in clean_data[user]:
                # NOTE: although there's some contexts with multiple responses, the next
                # line basically uses the last score for each context (so there is no
                # context overlap between train and test)
                if int(dic['c_id']) >= 0:
                    score_dic[dic['c_id']] = [dic['overall1'], dic['overall2'], \
                                              dic['overall3'], dic['overall4']]

        # Retrieve scores and valid context ids from clean_data.pkl
        valid_contextids = []
        context_list = []
        gtresponses = []
        model_responses = []
        scores = []
        model_names = []
        # Retrieve contexts and model responses from contexts.pkl
        # Each entry has the format [id, context, m1, m2, m3, m4]
        for c in contexts:
            # Check if we have seen this score.
            if c[0] in score_dic:
                if 'human' not in model_map[c[0]]:
                    continue
                if len(model_map[c[0]]) != 4:
                    continue
                valid_contextids.append(c[0])
                # TODO: Format these to remove html.
                context_list.append(c[1])
                model_responses.append(c[2:6])
                assert len(model_responses[-1]) == 4
                scores.append(score_dic[c[0]])
                assert len(scores[-1]) == 4
                gtresponses.append(gt_unordered[c[0]])
                model_names.append(model_map[c[0]])
                assert len(model_names[-1]) == 4

        # Flatten the lists.
        model_responses = [i for sublist in model_responses for i in sublist]
        scores = [float(i) for sublist in scores for i in sublist]
        model_names = [m for sublist in model_names for m in sublist]

        valid_contextids.sort()

        return context_list, gtresponses, model_responses, scores, valid_contextids, model_names

    def _preprocess(self, s):
        # This uses our generic preprocessor and adds additional preprocessing specific to this dataset.
        s = s.replace('<br />', '</s> ').replace('<first_speaker>', 'A:').replace('<second_speaker>', 'B:').replace( \
            '<third_speaker>', 'A:').replace('\n', '')
        return self.preprocessor.preprocess(s)

    def load_data(self):
        ''' This method will load the data from the AMT experiment.
          All data will be preprocessed into a standard format.
        '''
        # The data was collected in two AMT rounds.
        # See /home/ml/mnosew1/data/amt-adem/README.txt for a description of each file.
        fnames1 = ['%s/clean_data.pkl' % self.amt_folder,
                   '%s/contexts.pkl' % self.amt_folder,
                   '%s/true.txt' % self.amt_folder]
        fnames2 = ['%s/clean_data_new.pkl' % self.amt_folder,
                   '%s/contexts_new.pkl' % self.amt_folder,
                   '%s/true_new.txt' % self.amt_folder]
        model_map = self._fix_modelmaps()

        contexts, gt_responses, model_responses = [], [], []
        human_scores, valid_ids, model_names = [], [], []

        for f_data, f_contexts, f_gt in [fnames1, fnames2]:
            cs, gt_rs, m_rs, scores, valid_c_ids, m_names = self._get_twitter_data(f_data, f_contexts, f_gt, model_map)
            contexts += cs
            gt_responses += gt_rs
            model_responses += m_rs
            human_scores += scores
            model_names += m_names
            valid_ids.append(valid_c_ids)

        # Get the total list of contexts we are using.
        valid_context_ids = self._combine_contextids(valid_ids[0], valid_ids[1])
        model_map = self._filter_modelmap(model_map, valid_context_ids)

        # Combine and preprocess (remove html, etc.) both rounds of data.
        contexts = [self._preprocess(s) for s in contexts]
        gt_responses = [self._preprocess(s) for s in gt_responses]
        model_responses = [self._preprocess(s) for s in model_responses]

        # new_model_rs = []
        # new_gt_rs = []
        # for c, r, m in zip(contexts, gt_responses, model_responses):
        # 	new_model_rs.append(c + m[4:])
        # 	new_gt_rs.append(c + ' ' + r)

        dataset = []
        for i in range(0, len(contexts)):
            c, r_gt = contexts[i], gt_responses[i]
            m_names = model_names[i * 4:(i + 1) * 4]
            m_rs = model_responses[i * 4:(i + 1) * 4]
            scores = human_scores[i * 4:(i + 1) * 4]

            # r_gt = c + ' ' + r_gt

            new_m_rs = []
            for m in m_rs:
                new_m_rs.append(c + m[4:])
            om_rs = m_rs
            m_rs = new_m_rs

            entry = {'c': c, 'r_gt': r_gt, 'r_models': {}}
            # for n, r, s, omr in zip(m_names, m_rs, scores, om_rs):
            # 	entry['r_models'][n] = [r, s, len(omr)]
            # 	#print r, len(omr), len(r)
            for n, r, s in zip(m_names, om_rs, scores):
                entry['r_models'][n] = [r, s, len(r)]
            # print r, len(omr), len(r)
            dataset.append(entry)

        return dataset


class Preprocessor(object):

    def preprocess(self, s):
        while '@@ ' in s:
            s = s.replace('@@ ', '')

        utterance = s.replace('@user', '<at>').replace('&lt;unk&gt;', '<unk>').replace('&lt;heart&gt;',
                                                                                       '<heart>').replace(
            '&lt;number&gt;', '<number>').replace('  ', ' </s> ').replace('  ', ' ')
        # Make sure we end with </s> token
        utterance = utterance.replace('user', '<at>')
        utterance = utterance.replace('A:', '<first_speaker>')
        utterance = utterance.replace('B:', '<second_speaker>')
        utterance = utterance.replace('& lt', '<')
        utterance = utterance.replace('& gt', '>')
        utterance = utterance.replace('&lt;', '<')
        utterance = utterance.replace('&gt;', '>')
        utterance = utterance.replace('\'', ' \'')
        utterance = utterance.replace('"', ' " ')
        utterance = utterance.replace("'", " '")
        utterance = utterance.replace(";", " ")
        utterance = utterance.replace("`", " ")
        utterance = utterance.replace("..", ".")
        utterance = utterance.replace("..", ".")
        utterance = utterance.replace("..", ".")
        utterance = utterance.replace(",,", ",")
        utterance = utterance.replace(",,", ",")
        utterance = utterance.replace(",,", ",")
        utterance = utterance.replace('.', ' . ')
        utterance = utterance.replace('!', ' ! ')
        utterance = utterance.replace('?', ' ? ')
        utterance = utterance.replace(',', ' , ')
        utterance = utterance.replace('~', '')
        utterance = utterance.replace('-', ' - ')
        utterance = utterance.replace('*', ' * ')
        utterance = utterance.replace('(', ' ')
        utterance = utterance.replace(')', ' ')
        utterance = utterance.replace('[', ' ')
        utterance = utterance.replace(']', ' ')
        utterance = re.sub('[\s]+', ' ', utterance)
        utterance = utterance.replace('  ', ' ')
        utterance = utterance.replace('  ', ' ')
        s = utterance
        while '! ! ! !' in s:
            s = s.replace('! ! ! !', '! ! !')
        # s = utterance.replace('/', ' ')
        while s[-1] == ' ':
            s = s[0:-1]
        if not s[-5:] == ' </s>':
            s = s + ' </s>'
        return str(s)
