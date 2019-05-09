from .context import config, webcam, model, data
import unittest
import tensorflow as tf
import numpy as np

class simple_unittest(unittest.TestCase):

    @unittest.skip("Test this separately")
    def test_streamImage(self):
        print("test_streamImage...")

        wb = webcam.webcam()
        
        self.assertTrue(wb.streamImage(isTraining=True))
    
    @unittest.skip("Test this separately")
    def test_changeLabel(self):
        print("test_changeLabel...")

        wb = webcam.webcam()

        labelToChange = "test123"
        idToChange = "005"

        self.assertEqual(wb.changeLabel(labelToChange, idToChange),(labelToChange, idToChange))

    @unittest.skip("Test this separately")
    def test_loadData(self):
        print("test_loadData...")

        dt = data.data()

        self.assertGreater(len(dt.loadData(isTraining=True)[0]), 0)

    @unittest.skip("Test this separately")
    def test_createModel(self):
        print("test_createModel...")

        md = model.model()

        self.assertIsNotNone(md.createModel((20,20,3), 5))
    
    #@unittest.skip("Test this separately")
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
        md.training(train_dat, train_lbl, valid_dat, valid_lbl, epochs=config.EPOCH_NUM, 
                                                steps_per_epoch=config.STEPS_PER_EPOCH, 
                                                batch=config.BATCH_SIZE)

    @unittest.skip("Test this separately")
    def test_evaluation(self):
        print("test_evaluation...")

        md = model.model()
        dt = data.data()

        dt.setImgSize(config.WIDTH_SIZE, config.HEIGHT_SIZE, config.INPUT_CH)

        test_dat = np.array(dt.loadData(isTraining=False)[0])
        test_lbl = np.array(dt.loadData(isTraining=False)[1])

        test_dat_rd, test_lbl_rd = dt.randomize(test_dat, test_lbl)

        print("test_dat_rd", test_dat_rd.shape)

        md.createModel((config.INPUT_ROW,config.INPUT_COL,config.INPUT_CH), config.OUTPUT_CLASS)
        
        print("Loading weight start")
        md.load_weight()

        print("Prediction start")
        md.evaluation(test_dat_rd, test_lbl_rd)

    @unittest.skip("Test this separately")
    def test_predict(self):
        print("test_predict...")

        md = model.model()
        dt = data.data()

        dt.setImgSize(config.WIDTH_SIZE, config.HEIGHT_SIZE, config.INPUT_CH)

        test_dat = np.array(dt.loadData(isTraining=False)[0])
        test_lbl = np.array(dt.loadData(isTraining=False)[1])

        test_dat_rd, test_lbl_rd = dt.randomize(test_dat, test_lbl)

        print("test_dat_rd", test_dat_rd.shape)

        md.createModel((config.INPUT_ROW,config.INPUT_COL,config.INPUT_CH), config.OUTPUT_CLASS)
        
        print("Loading weight start")
        md.load_weight()

        print("Prediction start")
        print(md.predict(test_dat_rd))

    @unittest.skip("Test this separately")
    def test_streamImageForPredict(self):
        print("test_streamImageForPredict...")

        md = model.model()
        dt = data.data()
        wb = webcam.webcam()

        md.createModel((config.INPUT_ROW,config.INPUT_COL,config.INPUT_CH), config.OUTPUT_CLASS)
        
        print("Loading weight start")
        md.load_weight()

        print("Prediction start")
        print(wb.streamImageForPredict(md.predict, dt.resize, config.WIDTH_SIZE, config.HEIGHT_SIZE))

    @unittest.skip("Test this separately")
    def test_importXML(self):

        dt = data.data()

        dt.importXML()

if __name__ == '__main__':
    unittest.main()
