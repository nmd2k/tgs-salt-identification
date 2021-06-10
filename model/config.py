# Model config ======
RUN_NAME        = 'unetv1'
N_CLASSES       = 1
INPUT_SIZE      = 128
EPOCHS          = 10
LEARNING_RATE   = 0.0001
START_FRAME     = 16
DROP_RATE       = 0.5

# Data config =======
SAVE_PATH       = './model/'
DATA_PATH       = './data/'
IMAGE_PATH      = './train/images/'
MASK_PATH       = './train/masks/'

REAL_SIZE       = 101
RANDOM_SEED     = 42
VALID_RATIO     = 0.2
BATCH_SIZE      = 16
NUM_WORKERS     = 0
CLASSES         = {1:'salt'}

PAD_LEFT_TOP     = int((INPUT_SIZE - REAL_SIZE)/2)
PAD_RIGHT_BOTTOM = INPUT_SIZE - PAD_LEFT_TOP - REAL_SIZE