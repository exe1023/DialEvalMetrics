from utils.load_data_utils import *

class KeywordExtractor():
    def __init__(self, candi_keywords=None, idf_dict=None):
        self.candi_keywords = candi_keywords
        self.idf_dict = idf_dict

    @staticmethod
    def is_keyword_tag(tag):
        return tag.startswith('VB') or tag.startswith('NN') or tag.startswith('JJ')

    @staticmethod
    def cal_tag_score(tag):
        if tag.startswith('VB'):
            return 1.
        if tag.startswith('NN'):
            return 2.
        if tag.startswith('JJ'):
            return 0.5
        return 0.

    def is_candi_keyword(self, keyword):
        if keyword in self.candi_keywords:
            return True
        return False

    def idf_extract(self, string, con_kw=None):
        tokens = simp_tokenize(string)
        seq_len = len(tokens)
        tokens = pos_tag(tokens)
        source = kw_tokenize(string)
        candi = []
        result = []

        for i, (word, tag) in enumerate(tokens):
            score = self.cal_tag_score(tag)
            if not self.is_candi_keyword(source[i]) or score == 0.:
                continue
            if con_kw is not None and source[i] in con_kw:
                continue
            score *= source.count(source[i])
            score *= 1 / seq_len
            score *= self.idf_dict[source[i]]
            candi.append((source[i], score))
            if score > 0.15:
                result.append(source[i])
        return result

    def perserve_non_lemmatized_tokens_idf_extract(self, string, con_kw=None):
        tokens = simp_tokenize(string)
        seq_len = len(tokens)
        tokens = pos_tag(tokens)
        source = kw_tokenize(string)
        candi = []
        lemmatized_keywords = []
        non_lemmatized_keywords = []
        for i, (word, tag) in enumerate(tokens):
            score = self.cal_tag_score(tag)
            if not self.is_candi_keyword(source[i]) or score == 0.:
                continue
            if con_kw is not None and source[i] in con_kw:
                continue
            score *= source.count(source[i])
            score *= 1 / seq_len
            score *= self.idf_dict[source[i]]
            candi.append((source[i], score))
            if score > 0.15:
                lemmatized_keywords.append(source[i])
                non_lemmatized_keywords.append(word)
        return lemmatized_keywords, non_lemmatized_keywords

    def extract(self, string):
        tokens = simp_tokenize(string)
        tokens = pos_tag(tokens)
        source = kw_tokenize(string)
        kwpos_alters = []
        for i, (word, tag) in enumerate(tokens):
            if source[i] and self.is_keyword_tag(tag):
                kwpos_alters.append(i)
        kwpos, keywords = [], []
        for id in kwpos_alters:
            if self.is_candi_keyword(source[id]):
                keywords.append(source[id])
        return list(set(keywords))

    def candi_extract(self, string):
        tokens = simp_tokenize(string)
        tokens = pos_tag(tokens)
        source = kw_tokenize(string)
        keywords = []
        for i, (_, tag) in enumerate(tokens):
            if source[i] and self.is_keyword_tag(tag):
                keywords.append(source[i])
        return list(set(keywords))
