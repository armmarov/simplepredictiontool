from .context import config, webcam, model, data
import unittest
import tensorflow as tf
import numpy as np
import cv2

class simple_unittest(unittest.TestCase):

    def test_training(self):
        print("test_training...")

        md = model.model()
        dt = data.data()

        dt.setImgSize(config.WIDTH_SIZE, config.HEIGHT_SIZE, config.INPUT_CH)

        train_dat, valid_dat = dt.genImage();

        md.createModel((config.INPUT_ROW,config.INPUT_COL,config.INPUT_CH), config.OUTPUT_CLASS, config.MODEL_TYPE)
        md.training(train_dat, None, valid_dat, None, epochs=config.TRAIN_EPOCH_NUM, 
                                                steps_per_epoch=config.TRAIN_STEPS_PER_EPOCH, 
                                                batch=config.TRAIN_BATCH_SIZE, optimizer=config.TRAIN_OPTIMIZER,
                                                loss=config.TRAIN_LOSS)

if __name__ == '__main__':
    unittest.main()
