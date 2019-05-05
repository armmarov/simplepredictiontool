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
            #cap = self.camera.captureImage()
            ret, frame = self.camera.captureImage()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(320, 240, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

class MainApplication(QDialog):

    def __init__(self, parent=None, camera=None, model=None, data=None):

        super(MainApplication, self).__init__(parent)

        self.camera = camera
        self.cameraState = False

        self.data = data
        self.model = model

        self.d_lbl, self.d_ind, self.d_api, self.d_speed = self.data.importXML()

        # Start camera thread
        self.th = CamThread(camera=self.camera)
        self.th.changePixmap.connect(self.update_image)
        self.th.start()

        self.model.createModel((config.INPUT_ROW,config.INPUT_COL,config.INPUT_CH), config.OUTPUT_CLASS)
        self.data.setImgSize(config.WIDTH_SIZE, config.HEIGHT_SIZE, config.INPUT_CH)

        self.init_UI()
    
    @pyqtSlot(str)
    def update_training_status(self, status):

        print("Training status: " + status)

    @pyqtSlot(QImage)
    def update_image(self, img):

        self.camPrev.setPixmap(QPixmap.fromImage(img))
    
    def capture_image(self):

        status, path = self.camera.saveImage()

        print("Succesfully saved to " + path)

    def start_training(self):

        train_dat = np.array(self.data.loadData(isTraining=True)[0])
        train_lbl = np.array(self.data.loadData(isTraining=True)[1])

        validation_dat = np.array(self.data.loadData(isTraining=False)[0])
        validation_lbl = np.array(self.data.loadData(isTraining=False)[1])

        self.model.training(train_dat, train_lbl, validation_dat, validation_lbl, epochs=config.EPOCH_NUM, 
                                                steps_per_epoch=config.STEPS_PER_EPOCH, 
                                                batch=config.BATCH_SIZE)

    def load_weight(self):

        print("Load weight")
        self.model.load_weight()
        print("Load weight successful")

    def predict(self):

        #print("Start prediction")
        rval, img = self.camera.captureImage()
        if rval:
            dat = []
            dat.append(self.data.resize(img))
            cv2.imwrite("./test.jpg", self.data.resize(img))
            resize_img = np.array(dat)
            ret = self.model.predict(resize_img)

            if ret > 0:
                for i in range(0,len(self.d_ind)):
                    if int(self.d_ind[i]) == ret:
                        url = 'http://blynk-cloud.com/' + str(config.BLYNK_TOKEN) + '/update/' + str(self.d_api[i]) + '?value=' + str(self.d_speed[i])
                        print(url)
                        data = ''
                        response = requests.get(url, data)
                        print(response)
                        break
    
    def predict_cnt(self):

        print("Start prediction continuously")
        while(1):
            self.predict()
            tm.sleep(0.1)

    def selectionChange(self, i):

        self.camera.changeLabel(self.d_ind[i])

    def init_UI(self):
        
        appLabel = QLabel("Status : Working")
        
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
        startTrainBtn = QPushButton("Start Training")
        startTrainBtn.clicked.connect(self.start_training)

        # Widget for Testing Control
        loadBtn = QPushButton("Load")
        loadBtn.clicked.connect(self.load_weight)
        predictionBtn = QPushButton("Predict")
        predictionBtn.clicked.connect(self.predict)
        predictionCntBtn = QPushButton("Predict Continuously")
        predictionCntBtn.clicked.connect(self.predict_cnt)
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(appLabel)
        topLayout.addStretch(1)

        midLayout = QVBoxLayout()
        midLayout.addWidget(camPreviewLabel)
        midLayout.addWidget(self.camPrev)
        midLayout.addStretch(1)

        trainLabel = QHBoxLayout()
        trainLabel.addWidget(selectTrainingLabel)
        trainLabel.addWidget(self.labelComboBox)
        trainCtrlBtns = QHBoxLayout()
        trainCtrlBtns.addWidget(captureBtn)
        trainCtrlBtns.addWidget(startTrainBtn)
        trainLayout = QVBoxLayout()
        trainLayout.addLayout(trainLabel)
        trainLayout.addLayout(trainCtrlBtns)
        trainLayout.addStretch(1)
        trainCtrlGroup = QGroupBox("Training Control")
        trainCtrlGroup.setLayout(trainLayout)

        testCtrlBtns = QHBoxLayout()
        testCtrlBtns.addWidget(loadBtn)
        testCtrlBtns.addWidget(predictionBtn)
        testCtrlBtns.addWidget(predictionCntBtn)
        testLayout = QVBoxLayout()
        testLayout.addLayout(testCtrlBtns)
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

        self.setWindowTitle("Blynk Prediction Apps")

if __name__=="__main__":
    
    import sys

    camera = webcam.webcam()
    data = data.data()
    model = model.model()
    
    app = QApplication([])
    mainApps = MainApplication(camera=camera, model=model, data=data)
    mainApps.show()
    sys.exit(app.exec_())
