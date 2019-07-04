from .context import config, webcam, model, data
import unittest
import tensorflow as tf
import numpy as np
import cv2

class simple_unittest(unittest.TestCase):

    def test_evaluation(self):
        print("test_evaluation...")

        md = model.model()
        dt = data.data()

        dt.setImgSize(config.WIDTH_SIZE, config.HEIGHT_SIZE, config.INPUT_CH)

        train_dat, test_dat = dt.genImage()

        md.createModel((config.INPUT_ROW,config.INPUT_COL,config.INPUT_CH), config.OUTPUT_CLASS, config.MODEL_TYPE)

        md.compileModel(config.TRAIN_OPTIMIZER, config.TRAIN_LOSS)
        
        print("Loading weight start")
        md.load_weight()

        print("Prediction start")
        md.predict_gen(test_dat)
        

if __name__ == '__main__':
    unittest.main()
