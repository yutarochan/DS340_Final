'''
EVE: Emotion Vector Embedding - KDTree Decoder
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
from scipy.spatial import KDTree

class KDTDecoder:
    def __init__(self, model=None):
        # Extract Parameters and Corpus
        self.model = model
        self.corpus = list(model.keys())

        # Setup KDT Decoder
        data = list(map(lambda x: tuple(x), self.model.values()))
        self.dec = KDTree(data)

    def decode(self, emotion, k=1, csim=False):
        # Perform KDTree Search Query
        data = self.dec.query(emotion, k=k)

        if k == 1:
            if csim:
                return (self.corpus[data[1]], data[0])
            else:
                return self.corpus[data[1]]
        else:
            data = list(zip(list(data[1]), list(data[0])))
            if csim:
                return list(map(lambda x: (self.corpus[x[0]], x[1]), data))
            else:
                return list(map(lambda x: self.corpus[x[0]], data))
