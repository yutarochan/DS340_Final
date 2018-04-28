'''
EVE: Emotion Vector Encoder - Mean Vectorization Method
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import numpy as np

class MeanVector:
    def __init__(self, corpus=None, model=None):
        self.corpus = corpus
        self.cat_map = {i:[] for i in range(len(self.corpus))}
        self.emo_dict = None

    def fit(self, input, vals):
        # Aggregate Independent Dimensional Samples
        for i, cat in enumerate(input):
            for j, val in enumerate(cat):
                if val == 1: self.cat_map[j].append(vals[i])

        # Compute Mean per Discrete Category
        mean = []
        data = np.array([self.cat_map[i] for i in range(len(self.corpus))])
        for i in range(len(self.corpus)):
            mean.append(np.mean(np.array(data[i]), axis=0))

        # Mean Vector to Emotion Corpus
        self.emo_dict = {self.corpus[i] : mean[i] for i in range(len(self.corpus))}

    def encode(self, emotion):
        return self.emo_dict[emotion]

    def get_dict(self):
        return self.emo_dict
