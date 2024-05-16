from typing import List
import numpy as np


class AppraisalUtils:
    def __init__(self, lexicon, word_to_appraisal_grp, appraisal_grp_to_idx):
        self.word_to_appraisal_grp = word_to_appraisal_grp
        self.appraisal_grp_to_idx = appraisal_grp_to_idx
        self.lexicon = lexicon

    def appraisal_features(self, document: str) -> List[float]:
        res = np.zeros(len(self.lexicon))
        tokens = [word for word in document.split(" ")]
        count_appraisal_grps = 0
        for token in tokens:
            if token in self.word_to_appraisal_grp:
                res[self.appraisal_grp_to_idx[self.word_to_appraisal_grp[token]]] += 1
                count_appraisal_grps += 1
        # normalize features by the count of appraisal groups if the count != 0
        res = res / count_appraisal_grps if count_appraisal_grps != 0 else res
        return res

    def transform(self, text_arr: List[str]) -> np.array:
        document = text_arr[0]
        return self.appraisal_features(document)