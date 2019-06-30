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

        train_dat = np.array(dt.loadData(isTraining=True)[0])
        train_lbl = np.array(dt.loadData(isTraining=True)[1])

        valid_dat = np.array(dt.loadData(isTraining=False)[0])
        valid_lbl = np.array(dt.loadData(isTraining=False)[1])

        md.createModel((config.INPUT_ROW,config.INPUT_COL,config.INPUT_CH), config.OUTPUT_CLASS)
        md.training(train_dat, train_lbl, valid_dat, valid_lbl, epochs=config.TRAIN_EPOCH_NUM, 
                                                steps_per_epoch=config.TRAIN_STEPS_PER_EPOCH, 
                                                batch=config.TRAIN_BATCH_SIZE, optimizer=config.TRAIN_OPTIMIZER,
                                                loss=config.TRAIN_LOSS)

if __name__ == '__main__':
    unittest.main()
