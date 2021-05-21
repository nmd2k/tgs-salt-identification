# Model config ======
RUN_NAME        = 'unetv0'
N_CLASSES       = 1
INPUT_SIZE      = 512
EPOCHS          = 5
LEARNING_RATE   = 0.0001
START_FRAME     = 16

# Data config =======
SAVE_PATH       = './model/'
DATA_PATH       = './data/'
IMAGE_PATH      = './train/images/'
MASK_PATH       = './train/masks/'

RANDOM_SEED     = 42
VALID_RATIO     = 0.2
BATCH_SIZE      = 16
NUM_WORKERS     = 0
CLASSES         = ['salt']
