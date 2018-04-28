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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

import params
import util.data_csv
from util.data_csv import EMOTICData

# Original Codebase
# https://www.kaggle.com/abnera/transfer-learning-keras-xception-cnn

'''
TODO:
- Implement callback mechanisms for model saving and weight decay.
- Early stopping mechanism(?)
'''

# Application Parameters
SEED = 9892
np.random.seed(seed=SEED)
tf.set_random_seed(seed=SEED)

# Model Training Parameters
EVE_DIM = 150                       # EVE Dimensions
IM_WIDTH, IM_HEIGHT = 255, 255      # Image Width and Height
BATCH_SIZE = 32                     # Minibatch Size
NB_EPOCHS = 50                      # Number of Epochs to Train Model
LEARN_RATE = 1e-4                   # SGD Learning Rate
MOMENTUM = 0.9                      # SGD Momentum to Avoid Local Minimum
BMLBLN = 126                        # alue is based on based model selected

''' Load Dataset '''
train_datagen = ImageDataGenerator(rescale=1. / 255)

train_data = EMOTICData(params.DATA_DIR, params.ANNOT_DIR_TRAIN, params.EVE_MODEL_DIR)
valid_data = EMOTICData(params.DATA_DIR, params.ANNOT_DIR_VALID, params.EVE_MODEL_DIR)

''' Setup Model '''
# Setup XceptionNet EMOTIC Model
base_model = Xception(input_shape=params.IM_DIM, weights='imagenet', include_top=False)

# Define Model Top Block
x = base_model.output
x = GlobalAveragePooling2D()(x)
pred = Dense(EVE_DIM)(x)

# Build Model
model = Model(base_model.input, pred)
# model.summary()

# Freeze Base Model Layer - Train Top Block
for layer in base_model.layers: layer.trainable = False

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

''' Train Model '''
# Train Top Layer Only
model.fit_generator(train_data.data_gen(BATCH_SIZE), samples_per_epoch=len(train_data), nb_epoch=NB_EPOCHS)

# Fine Tune Model
# model.load_weights(top_weights_path) # Use Best Weights from Training

# BMLBLN Training Activation
for layer in model.layers[:BMLBLN]:
    layer.trainable = False
for layer in model.layers[BMLBLN:]:
    layer.trainable = True

# Recompile Model Again...
model.compile(optimizer='adam', loss='mean_squared_error')

# Fine Tune Model
model.fit_generator(train_data.data_gen(BATCH_SIZE), samples_per_epoch=len(train_data), nb_epoch=NB_EPOCHS)
