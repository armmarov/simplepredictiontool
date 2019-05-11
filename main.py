from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from tensorflow import Graph
from libs import webcam, model, data, config
import cv2
import numpy as np
import requests
import time as tm

class CamThread(QThread):

    changePixmap = pyqtSignal(QImage)

    def __init__(self, parent=None, camera=None):
        QThread.__init__(self, parent=parent)
        print("[CamThread] Thread started..")
        self.camera = camera

    def run(self):
        print("[CamThread] Run")
        while True:
            ret = self.camera.captureImage(config.WIDTH_SIZE, config.HEIGHT_SIZE)
            if ret:
                img = self.camera.getImage()
                rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(320, 240, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

class MLThread(QThread):

    MLStatus = pyqtSignal(str)

    def __init__(self, parent=None, camera=None, data=None):
        QThread.__init__(self, parent=parent)
        print("[MLThread] Thread started..")
        self.camera = camera
        self.model = model.model()

        self.data = data
        self.mode = "none"

        self.cont = False

        self.mlModel = self.model.createModel((config.INPUT_ROW,config.INPUT_COL,config.INPUT_CH), config.OUTPUT_CLASS, config.MODEL_TYPE)
        self.data.setImgSize(config.WIDTH_SIZE, config.HEIGHT_SIZE, config.INPUT_CH)

        self.d_lbl, self.d_ind, self.d_api = self.data.importXML()

        self.sens_cnt = 0
        self.prev_num = 0

    def run(self):
        print("[MLThread] Run ", self.mode)

        if self.mode == "train":
            self.train()
        elif self.mode == "load":
            self.load()
        elif self.mode == "predict":
            self.predict(False)
        elif self.mode == "predict_cnt":
            self.predict_cnt()
    
    def train(self):
        train_dat = np.array(self.data.loadData(isTraining=True)[0])
        train_lbl = np.array(self.data.loadData(isTraining=True)[1])

        validation_dat = np.array(self.data.loadData(isTraining=False)[0])
        validation_lbl = np.array(self.data.loadData(isTraining=False)[1])

        self.model.training(train_dat, train_lbl, validation_dat, validation_lbl, epochs=config.EPOCH_NUM, 
                                                steps_per_epoch=config.STEPS_PER_EPOCH, 
                                                batch=config.BATCH_SIZE)
        
        self.MLStatus.emit("TRAINSUCCESS")
    
    def load(self):
        print("Loading weights")
        ret = self.model.load_weight()
        print("Finish loading weights")
        
        if ret != None:
            self.MLStatus.emit("LOADSUCCESS")
    
    def predict(self, isCont):
        #print("Start prediction")
        imgCrop = self.camera.getCropImage()

        ret = self.model.predict(imgCrop)

        if ret > 0 and ret == self.prev_num:
            if self.sens_cnt >= config.PRED_SENSITIVITY:  
                self.sens_cnt = 0
                for i in range(0,len(self.d_ind)):
                    if int(self.d_ind[i]) == ret:
                        url = str(self.d_api[i])
                        print(url)
                        data = ''
                        response = requests.get(url, data)
                        print(response)
                        break
            else :
                self.sens_cnt = self.sens_cnt + 1
        else:
            self.prev_num = ret

        if isCont == False:
            self.MLStatus.emit("PREDSUCCESS")

    def predict_cnt(self):

        while(self.cont):
            if self.camera.getNewPict():
                self.predict(True)
            tm.sleep(config.PRED_CONT_DELAY)
        
        print("Continuous Prediction Finished")

class MainApplication(QDialog):

    def __init__(self, parent=None, camera=None, data=None):

        super(MainApplication, self).__init__(parent)

        self.camera = camera
        self.cameraState = False

        self.data = data
        #self.model = model

        self.d_lbl, self.d_ind, self.d_api = self.data.importXML()

        # Start camera thread
        self.th = CamThread(camera=self.camera)
        self.th.changePixmap.connect(self.update_image)
        self.th.start()

        # Start ML thread
        self.th_ml = MLThread(camera=self.camera, data=self.data)
        self.th_ml.MLStatus.connect(self.update_ml)

        self.init_UI()
    
    @pyqtSlot(str)
    def update_training_status(self, status):

        print("Training status: " + status)

    @pyqtSlot(QImage)
    def update_image(self, img):

        self.camPrev.setPixmap(QPixmap.fromImage(img))
    
    @pyqtSlot(str)
    def update_ml(self, stat):

        if stat == "LOADSUCCESS":
            self.predictionBtn.setEnabled(True)
            self.predictionCntBtn.setEnabled(True)
            self.appLabel.setText("Status : Idle")

        elif stat == "PREDSUCCESS":
            self.loadBtn.setEnabled(True)
            self.predictionBtn.setEnabled(True)
            self.predictionCntBtn.setEnabled(True)
            self.appLabel.setText("Status : Idle")
        
        elif stat == "TRAINSUCCESS":
            self.startTrainBtn.setEnabled(True)
            self.loadBtn.setEnabled(True)
            self.predictionBtn.setEnabled(True)
            self.predictionCntBtn.setEnabled(True)
            self.appLabel.setText("Status : Idle")
    
    def capture_image(self):

        status, path = self.camera.saveImage()

        print("Succesfully saved to " + path)

    def start_training(self):

        self.startTrainBtn.setEnabled(False)
        self.loadBtn.setEnabled(False)
        self.predictionBtn.setEnabled(False)
        self.predictionCntBtn.setEnabled(False)
        self.appLabel.setText("Status : Training...")
        self.th_ml.mode = "train"
        self.th_ml.start()

    def load_weight(self):

        self.appLabel.setText("Status : Loading Weights...")
        self.th_ml.mode = "load"
        self.th_ml.start()

    def predict(self):

        self.loadBtn.setEnabled(False)
        self.predictionBtn.setEnabled(False)
        self.predictionCntBtn.setEnabled(False)
        self.appLabel.setText("Status : Predicting...")

        self.th_ml.mode = "predict"
        self.th_ml.start()
    
    def predict_cnt(self):

        self.loadBtn.setEnabled(False)
        self.predictionBtn.setEnabled(False)
        self.predictionCntBtn.setEnabled(False)
        self.predictionCntStopBtn.setEnabled(True)
        self.appLabel.setText("Status : Continuously predicting...")

        self.th_ml.cont = True
        self.th_ml.mode = "predict_cnt"
        self.th_ml.start()

    def predict_cnt_stop(self):

        self.th_ml.cont = False

        self.loadBtn.setEnabled(True)
        self.predictionBtn.setEnabled(True)
        self.predictionCntBtn.setEnabled(True)
        self.predictionCntStopBtn.setEnabled(False)
        self.appLabel.setText("Status : Idle")

    def selectionChange(self, i):

        self.camera.changeLabel(self.d_ind[i])
    
    def change_capture_state(self, btn):
        #print("Change state ", btn.text())
        if btn.text() == "For Training":
            self.camera.setTrainingMode(True)
        elif btn.text() == "For Validation":
            self.camera.setTrainingMode(False)

    def clearDatasets(self):

        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Warning)
        self.msgBox.setText("Are you sure you want to delete all datasets?")
        self.msgBox.setInformativeText("This action cannot be undo.")
        self.msgBox.setWindowTitle("Removing Training Datasets")
        self.msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        self.msgBox.buttonClicked.connect(self.removeDatasets)

        retval = self.msgBox.exec_()
    
    def removeDatasets(self, i):
        print(i.text())
        if i.text() == "&OK":
            self.data.removeDatasets()
            print("Done removing")

    def init_UI(self):
        
        self.appLabel = QLabel("Status : Idle")
        
        # Camera Preview
        camPreviewLabel = QLabel("Camera Preview")
        self.camPrev = QLabel()
        self.camPrev.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.camPrev.setFixedSize(320,240)

        # Widget for Training Control
        selectTrainingLabel = QLabel("Select Label:")
        self.labelComboBox = QComboBox()
        
        for i in range(0, len(self.d_lbl)):
                self.labelComboBox.addItem(self.d_lbl[i])
        self.labelComboBox.currentIndexChanged.connect(self.selectionChange)

        captureBtn = QPushButton("Capture")
        captureBtn.clicked.connect(self.capture_image)
        self.trainRadioBtn = QRadioButton("For Training")
        self.trainRadioBtn.toggled.connect(lambda:self.change_capture_state(self.trainRadioBtn))
        self.trainRadioBtn.setChecked(True)
        self.validRadioBtn = QRadioButton("For Validation")
        self.validRadioBtn.toggled.connect(lambda:self.change_capture_state(self.validRadioBtn))
        self.startTrainBtn = QPushButton("Start Training")
        self.startTrainBtn.clicked.connect(self.start_training)
        self.clearBtn = QPushButton("Clear Data")
        self.clearBtn.clicked.connect(self.clearDatasets)        

        # Widget for Testing Control
        self.loadBtn = QPushButton("Load")
        self.loadBtn.clicked.connect(self.load_weight)
        self.predictionBtn = QPushButton("Predict")
        self.predictionBtn.clicked.connect(self.predict)
        self.predictionBtn.setEnabled(False)
        self.predictionCntBtn = QPushButton("Predict Cont")
        self.predictionCntBtn.clicked.connect(self.predict_cnt)
        self.predictionCntBtn.setEnabled(False)
        self.predictionCntStopBtn = QPushButton("Stop")
        self.predictionCntStopBtn.clicked.connect(self.predict_cnt_stop) 
        self.predictionCntStopBtn.setEnabled(False)
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.appLabel)
        topLayout.addStretch(1)

        midLayout = QVBoxLayout()
        midLayout.addWidget(camPreviewLabel)
        midLayout.addWidget(self.camPrev)
        midLayout.addStretch(1)

        trainLabel = QHBoxLayout()
        trainLabel.addWidget(selectTrainingLabel)
        trainLabel.addWidget(self.labelComboBox)
        trainCtrlBtns = QHBoxLayout()
        trainCtrlBtns.addWidget(self.trainRadioBtn)
        trainCtrlBtns.addWidget(self.validRadioBtn)   
        trainCtrlBtns1 = QHBoxLayout()
        trainCtrlBtns1.addWidget(self.clearBtn)
        trainCtrlBtns1.addWidget(captureBtn)
        trainCtrlBtns2 = QHBoxLayout()
        trainCtrlBtns2.addWidget(self.startTrainBtn)
        trainLayout = QVBoxLayout()
        trainLayout.addLayout(trainLabel)
        trainLayout.addLayout(trainCtrlBtns)
        trainLayout.addLayout(trainCtrlBtns1)
        trainLayout.addLayout(trainCtrlBtns2)
        trainLayout.addStretch(1)
        trainCtrlGroup = QGroupBox("Training Control")
        trainCtrlGroup.setLayout(trainLayout)

        testCtrlBtns = QHBoxLayout()
        testCtrlBtns.addWidget(self.loadBtn)
        testCtrlBtns.addWidget(self.predictionBtn)
        testCtrl1Btns = QHBoxLayout()
        testCtrl1Btns.addWidget(self.predictionCntBtn)
        testCtrl1Btns.addWidget(self.predictionCntStopBtn)
        testLayout = QVBoxLayout()
        testLayout.addLayout(testCtrlBtns)
        testLayout.addLayout(testCtrl1Btns)
        testLayout.addStretch(1)
        testCtrlGroup = QGroupBox("Testing Control")
        testCtrlGroup.setLayout(testLayout)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addLayout(midLayout, 1, 0, 1, 2)
        mainLayout.addWidget(trainCtrlGroup, 2, 0, 1, 2)
        mainLayout.addWidget(testCtrlGroup, 3, 0, 1, 2)
        mainLayout.setRowStretch(1,1)
        mainLayout.setRowStretch(2,1)
        mainLayout.setColumnStretch(0,1)
        mainLayout.setColumnStretch(1,1)
        
        self.setLayout(mainLayout)

        self.setWindowTitle("Blynk Controller Apps")

if __name__=="__main__":
    
    import sys

    camera = webcam.webcam()
    data = data.data()
    #model = model.model()
    
    app = QApplication([])
    mainApps = MainApplication(camera=camera, data=data)
    mainApps.show()
    sys.exit(app.exec_())
