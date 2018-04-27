'''
EVE: Emotion Vector Encoder - Word Embedding Method
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import gensim
import numpy as np
from gensim.models.word2vec import Word2Vec

class WordEmbedding:
    def __init__(self, corpus=None, model=None):
        self.corpus = corpus
        if model:
            self.model = Word2Vec.load(model)
        else:
            self.model = None

    def fit(self, dataset, dim=150, epochs=15, window=2, min_count=2, threads=4):
        self.model = Word2Vec(dataset, size=dim, window=window, min_count=min_count, workers=threads, sg=1)
        self.model.train(dataset, total_examples=model.corpus_count, epochs=epochs)

    def encode(self, emotion):
        return self.model[emotion]

    def decode(self, vector, k=1, csim=False):
        data = self.model.most_similar(positive=[vector], topn=k)
        if k == 1:
            if csim: return data[0]
            else: return data[0][0]
        else:
            if csim: return data
            else: return list(map(lambda x: x[0], data))
