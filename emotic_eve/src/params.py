'''
EMOTIC CNN Baseline: Parameter Configuration
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function

''' APPLICATION PARAMETERS '''
ROOT_DIR = '/storage/home/yjo5006/work/emotic_data/'
DATA_DIR = ROOT_DIR + '/emotic/'
MODEL_DIR = ROOT_DIR + '/model/'
LOSS_LOG_DIR = ROOT_DIR + '/loss_log/'

ANNOT_DIR_TRAIN = ROOT_DIR + '/emotic/train_annot.csv'
ANNOT_DIR_VALID = ROOT_DIR + '/emotic/valid_annot.csv'
ANNOT_DIR_TEST  = ROOT_DIR + '/emotic/test_annot.csv'

EVE_MODEL_DIR = '/storage/home/yjo5006/DS340_Final/data/eve/emotic_w2v.pickle'

MODEL_DIR = ROOT_DIR + '/models/'

REPORT_FREQ = 10
USE_CUDA = True
NUM_WORKERS = 32

VISTORCH_LOG = False

''' APPLICATION CONSTANTS '''
NDIM_DISC = 26
NDIM_CONT = 3

IM_DIM = (256, 256, 3)

''' TRAINING PARAMETERS '''
EMOTIC_MEAN = [0.45897533113801, 0.44155118600299, 0.40552199274783]
EMOTIC_STD  = [0.23027497714954, 0.22752317402935, 0.23638979553161]

BN_EPS = 0.001
STD_VAR_INIT = 1e-2
TRAIN_LR = 1e-6

TRAIN_BATCH_SIZE = 100
TRAIN_DATA_SHUFFLE = True

VALID_BATCH_SIZE = 100
VALID_DATA_SHUFFLE = True

START_EPOCH = 1
TRAIN_EPOCH = 150

SAVE_FREQ = 15

# Loss Parameters
W_CONT = 1.0
W_DISC = 1.0/6.0
LOSS_CONT_MARGIN = 0.1
LDISC_C = 1.2

''' TESTING PARAMETERS '''
TEST_BATCH_SIZE = 100
