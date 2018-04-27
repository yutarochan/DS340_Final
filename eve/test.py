'''
EVE: Emotion Vector Encoder
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import pandas as pd

from eve.encode.mvm import MeanVector
from eve.encode.w2v import WordEmbedding
from eve.decode.kdt import KDTDecoder

''' Mean Vectorization Method Test '''
header = ['id', 'user_id', 'video_id', 'entity_id', 'is_corrupted', 'emo_cat',
            'valence', 'arousal', 'dominance', 'gender', 'age', 'ethnicity',
            'st_frame', 'ed_frame', 'eval_time', 'qc_pass']

data = pd.read_csv('data/annotations.csv', header=None, names=header)

# Extract Values
emo_cat = list(data.emo_cat.values)
valence = list(data.valence.values)
arousal = list(data.arousal.values)
dominance = list(data.dominance.values)

# Setup as List
emo_cat = [[int(e) for e in emo] for emo in emo_cat]
vad = [[valence[i], arousal[i], dominance[i]] for i in range(len(valence))]

# Core Emotion Corpus
emotions = ['Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
            'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
            'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue',
            'Embarrassment', 'Yearning', 'Disapproval', 'Aversion', 'Annoyance',
            'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain', 'Suffering']

'''
# Test EVE MVM Encoder
vec = MeanVector(emotions)
vec.fit(emo_cat, vad)

print(vec.encode('Anger'))

# Setup EVE KDT Decoder
dec = KDTDecoder(vec.get_dict())

# Test on Input
peace_neg = vec.encode('Peace')
print(dec.decode(peace_neg, k=5, csim=False))
'''

''' Word Embedding Method Test '''
# Test EVE W2V Training Process
# dataset = [[emotions[i] for i, e in enumerate(emo) if e == '1'] for emo in emo_cat]

# Test EVE W2V Encoder - Load Pretrained Model
vec = WordEmbedding(corpus=emotions, model='data/bold_w2v.pickle')
# print(vec.encode('Peace'))
print(vec.decode(vec.encode('Happiness') + vec.encode('Surprise'), k=10, csim=False))
