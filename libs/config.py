
## Image size
WIDTH_SIZE  = 96   
HEIGHT_SIZE = 96

## Neural network configuration
OUTPUT_CLASS = 6
INPUT_ROW    = HEIGHT_SIZE
INPUT_COL    = WIDTH_SIZE
INPUT_CH     = 1  # 3 for RGB, 1 for BLACK/WHITE
#MODEL_TYPE   = "mobilenetv2"
MODEL_TYPE   = "ud1"

## Training parameters
EPOCH_NUM       = 1000
STEPS_PER_EPOCH = 5
BATCH_SIZE      = 10

## Prediction
PRED_SENSITIVITY  = 1
PRED_CONT_DELAY   = 0.5