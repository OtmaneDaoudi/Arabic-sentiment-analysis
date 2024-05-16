from typing import List
import numpy as np

class AppraisalUtils:
    def __init__(self, lexicon, word_to_appraisal_grp, appraisal_grp_to_idx, bow_vectorizer):
        self.word_to_appraisal_grp = word_to_appraisal_grp
        self.appraisal_grp_to_idx = appraisal_grp_to_idx
        self.lexicon = lexicon
        self.bow_vectorizer = bow_vectorizer

    def appraisal_features(self, document: str) -> List[float]:
        res = np.zeros((1, len(self.lexicon)))
        tokens = [word for word in document.split(' ')]
        count_appraisal_grps = 0
        for token in tokens:
            if token in self.word_to_appraisal_grp:
                res[0, self.appraisal_grp_to_idx[self.word_to_appraisal_grp[token]]] += 1
                count_appraisal_grps += 1
        res = res / count_appraisal_grps if count_appraisal_grps != 0 else res
        return res

    def transform(self, text_arr: List[str]) -> np.array:
        document = text_arr[0]
        bow_features = self.bow_vectorizer.transform([document]).toarray()
        appraisal_features_ = self.appraisal_features(document)
        return np.hstack((bow_features, appraisal_features_))