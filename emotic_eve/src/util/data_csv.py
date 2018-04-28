'''
EMOTIC Dataset Utility [CSV Version]
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import cv2
import sys
import time
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
from PIL import ImageFile
from functools import reduce
from multiprocessing import Pool

# from util.eve.encode.w2v import WordEmbedding
import params
from util.eve.encode.w2v import WordEmbedding

# FIX: Truncated Image Error
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO: Implement multithreaded GPU import data generator.
# https://www.sagivtech.com/2017/09/19/optimizing-pytorch-training-code/
# https://www.kaggle.com/danielhavir/pytorch-dataloader

# TODO: Implement thread safe method for concurrency handling and data IO pipelining to improve data load procedure to mask behind GPU.

# Emotion Categories
cat_name = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion',
            'Confidence', 'Disapproval', 'Disconnection', 'Disquietment',
            'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
            'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace',
            'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
            'Sympathy', 'Yearning']

class EMOTICData:
    def __init__(self, root_dir, annotations, eve_model):
        # Extract Parameters
        self.ROOT_DIR = root_dir
        self.ANOT_DIR = annotations
        self.EVE_DIR = eve_model

        # Load Annotation File
        start = time.time()
        self.annot = pd.read_csv(self.ANOT_DIR, skiprows=0).dropna()
        end = time.time()

        # Print Statement
        print('LOADED', self.ANOT_DIR, '\t[', len(self.annot),']\t', (end - start), ' sec.')

        # Load EVE Model
        self.eve = WordEmbedding(corpus=cat_name, model=self.EVE_DIR)

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, index):
        # Load Image File
        filename = self.annot.iloc[index, 0]

        # Extract Image
        # bb = self.annot.iloc[index, [1,2,3,4]].tolist()
        image = np.array(Image.open(open(filename, 'rb')).convert('RGB'))
        image = cv2.resize(image, (256, 256))
        # body = image.crop((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))

        # Extract and Encode Label
        category = self.annot.iloc[index, range(5, 31)].astype(int).tolist()
        cat_vecs = [self.eve.encode(cat_name[i]) for i, cat in enumerate(category) if cat == 1]
        emo_vecs = reduce(np.add, cat_vecs)

        return (image, emo_vecs)

    def data_gen(self, batch_size):
        X = []
        y = []
        for i in range(len(self)): 
            X.append(self[i][0])
            y.append(self[i][1])
            
            if i != 0 and i % batch_size == 0:
                yield (np.array(X), np.array(y))
                X = []
                y = []

if __name__ == '__main__':
    ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'
    ANOT_DIR = ROOT_DIR + '/emotic/train_annot.csv'
    EVE_DIR = '/storage/home/yjo5006/DS340_Final/data/eve/emotic_w2v.pickle'

    # Load Training Data
    # start = time.time()
    data = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, EVE_DIR)
    print(data[325])
    # end = time.time()
    # print('Train - Total Time Elapsed: ' + str(end - start) + ' sec.')

    # sample = data[0]
    '''
    # Load Validation Data
    start = time.time()
    data = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, 'val')
    end = time.time()
    print('Validation - Total Time Elapsed: ' + str(end - start) + ' sec.')

    # Load Testing Data
    start = time.time()
    data = EMOTICData(ROOT_DIR+'emotic/', ANOT_DIR, 'test')
    end = time.time()
    print('Validation - Total Time Elapsed: ' + str(end - start) + ' sec.')
    '''

    # Initialize Data Loader
    # data_loader = torch.utils.data.DataLoader(data)

    # Dataset Batch Loading
    # for batch_idx, sample in enumerate(data_loader):
        # image = Variable(sample[0])
        # print(sample[3])

        # print(len(sample))
        # print(sample[batch_idx][3])

    # print(data[263]) # Test Load
