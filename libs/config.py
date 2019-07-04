
## Image size
WIDTH_SIZE  = 96   
HEIGHT_SIZE = 96

## Neural network configuration
OUTPUT_CLASS = 2
INPUT_ROW    = HEIGHT_SIZE
INPUT_COL    = WIDTH_SIZE
INPUT_CH     = 3  # 3 for RGB, 1 for BLACK/WHITE
#MODEL_TYPE   = "mobilenetv2"
#MODEL_TYPE   = "ud1"
MODEL_TYPE   = "ud2"

## Training parameters
TRAIN_OPTIMIZER       = 'adam'
#TRAIN_LOSS            = 'sparse_categorical_crossentropy'
TRAIN_LOSS            = 'categorical_crossentropy'
TRAIN_EPOCH_NUM       = 200
TRAIN_STEPS_PER_EPOCH = 5
TRAIN_BATCH_SIZE      = 10

## Prediction
PRED_SENSITIVITY  = 1
PRED_CONT_DELAY   = 0.5
