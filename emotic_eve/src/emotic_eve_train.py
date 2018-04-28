'''
EMOTIC Model using EVE Representation - Training Routine
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import backend as k
from keras.optimizers import *
from keras.models import Model
from keras.applications import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

import util.data_csv

# Original Codebase
# https://www.kaggle.com/abnera/transfer-learning-keras-xception-cnn

# Application Parameters
SEED = 9892
np.random.seed(seed=SEED)
tf.set_random_seed(seed=SEED)

# Model Training Parameters
EVE_DIM = 150                       # EVE Dimensions
IM_WIDTH, IM_HEIGHT = 255, 255      # Image Width and Height
BATCH_SIZE = 32                     # Minibatch Size
NB_EPOHCS = 50                      # Number of Epochs to Train Model
LEARN_RATE = 1e-4                   # SGD Learning Rate
MOMENTUM = 0.9                      # SGD Momentum to Avoid Local Minimum

# Setup XceptionNet EMOTIC Model
base_model = Xception(input_shape=(IM_WIDTH, IM_HEIGHT, 3), weights='imagenet', include_top=False)

# Define Model Top Block
x = base_model.output
x = GlobalAveragePooling2D()(x)
pred = Dense(EVE_DIM)(x)

# Build Model
model = Model(base_model.input, pred)
model.summary()

# Freeze Base Model Layer - Train Top Block
for layer in base_model.layers: layer.trainable = False

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')
