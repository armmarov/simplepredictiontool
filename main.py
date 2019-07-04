"""
Simple Prediction Tool is developed to control the IoT Blynk device.
Developed by Ammar (armmarov@gmail.com)
"""
import time as tm
import cv2
import numpy as np
import requests
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QDialog, QApplication, QFrame, QRadioButton, QHBoxLayout, QVBoxLayout, QGroupBox, QGridLayout, QAction, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from libs import webcam, model, data, config

class CamThread(QThread):
    """ A separate thread for camera processing """

    changePixmap = pyqtSignal(QImage)

    def __init__(self, parent=None, camera=None):

        QThread.__init__(self, parent=parent)
        print("[CamThread] Thread started..")
        self.camera = camera

        self.capture = True

    def run(self):
        """ Thread main function

        Parameters:

        Return:

        """
        print("[CamThread] Run")
        while self.capture:
            ret = self.camera.captureImage(config.WIDTH_SIZE, config.HEIGHT_SIZE)
            if ret:
                img = self.camera.getImage()
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                convert_to_qtformat = QImage(rgb_img.data,
                                             rgb_img.shape[1],
                                             rgb_img.shape[0],
                                             QImage.Format_RGB888)
                _p = convert_to_qtformat.scaled(320, 240, Qt.KeepAspectRatio)
                self.changePixmap.emit(_p)

        print("[CamThread] Stop")

class MLThread(QThread):
    """ A separate thread for machine learning processing """

    MLStatus = pyqtSignal(str)

    def __init__(self, parent=None, camera=None, data_mdl=None):

        QThread.__init__(self, parent=parent)
        print("[MLThread] Thread started..")
        self.camera = camera
        self.model = model.model()

        self.data = data_mdl
        self.mode = "none"

        self.cont = False

        self.ml_model = self.model.createModel((config.INPUT_ROW,
                                                config.INPUT_COL,
                                                config.INPUT_CH),
                                               config.OUTPUT_CLASS,
                                               config.MODEL_TYPE)

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

        train_dat, validation_dat = self.data.genImage()

        if len(train_dat) == 0 or len(train_lbl) == 0 or len(validation_dat) == 0 or len(validation_lbl) == 0:
            self.MLStatus.emit("INPUTEMPTY")
            return

        self.model.training(train_dat, None, validation_dat, None,

                            epochs=config.TRAIN_EPOCH_NUM,
                            steps_per_epoch=config.TRAIN_STEPS_PER_EPOCH,
                            batch=config.TRAIN_BATCH_SIZE,
                            optimizer=config.TRAIN_OPTIMIZER,
                            loss=config.TRAIN_LOSS)

        self.MLStatus.emit("TRAINSUCCESS")

    def load(self):

        print("Compile model")
        self.model.compileModel(config.TRAIN_OPTIMIZER, config.TRAIN_LOSS)

        print("Loading weights")
        ret = self.model.load_weight()
        print("Finish loading weights")

        if ret is not None:
            self.MLStatus.emit("LOADSUCCESS")

    def predict(self, is_cont):

        #print("Start prediction")
        img_crop = self.camera.getCropImage()

        if img_crop is None:
            self.MLStatus.emit("INPUTEMPTY")
            return

        ret = self.model.predict(img_crop)

        if ret > 0 and ret == self.prev_num:
            if self.sens_cnt >= config.PRED_SENSITIVITY:
                self.sens_cnt = 0
                for i in range(0, len(self.d_ind)):
                    if int(self.d_ind[i]) == ret:
                        url = str(self.d_api[i])
                        print(url)
                        response = requests.get(url, '')
                        print(response)
                        break
            else:
                self.sens_cnt = self.sens_cnt + 1
        else:
            self.prev_num = ret

        if not is_cont:
            self.MLStatus.emit("PREDSUCCESS")

    def predict_cnt(self):

        while self.cont:
            if self.camera.getNewPict():
                self.predict(True)
            tm.sleep(config.PRED_CONT_DELAY)

        print("Continuous Prediction Finished")

class MainApplication(QDialog):
    """ Class for main application """

    def __init__(self, parent=None, camera=None, data_mdl=None):

        super(MainApplication, self).__init__(parent)

        self.camera = camera
        self.data = data_mdl

        self.d_lbl, self.d_ind, self.d_api = self.data.importXML()

        # Start camera thread
        self.th_im = CamThread(camera=self.camera)
        self.th_im.changePixmap.connect(self.update_image)
        self.th_im.start()

        # Start ML thread
        self.th_ml = MLThread(camera=self.camera, data_mdl=self.data)
        self.th_ml.MLStatus.connect(self.update_ml)

        self.init_ui()

    @pyqtSlot(QImage)
    def update_image(self, img):

        self.cam_prev.setPixmap(QPixmap.fromImage(img))

    @pyqtSlot(str)
    def update_ml(self, stat):

        if stat == "LOADSUCCESS":
            self.prediction_btn.setEnabled(True)
            self.prediction_cnt_btn.setEnabled(True)
            self.app_label.setText("Status : Idle")

        elif stat == "PREDSUCCESS":
            self.load_btn.setEnabled(True)
            self.prediction_btn.setEnabled(True)
            self.prediction_cnt_btn.setEnabled(True)
            self.app_label.setText("Status : Idle")

        elif stat == "TRAINSUCCESS":
            self.start_train_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.prediction_btn.setEnabled(True)
            self.prediction_cnt_btn.setEnabled(True)
            self.app_label.setText("Status : Idle")
        
        elif stat == "INPUTEMPTY":
            self.start_train_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.prediction_btn.setEnabled(True)
            self.prediction_cnt_btn.setEnabled(True)
            print("Input Empty !!!")
            self.app_label.setText("Status : Idle")

    def capture_image(self):

        status, path = self.camera.saveImage()

        print(status + " saved to " + path)

    def start_training(self):

        self.start_train_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.prediction_btn.setEnabled(False)
        self.prediction_cnt_btn.setEnabled(False)
        self.app_label.setText("Status : Training...")
        self.th_ml.mode = "train"
        self.th_ml.start()

    def load_weight(self):

        self.app_label.setText("Status : Loading Weights...")
        self.th_ml.mode = "load"
        self.th_ml.start()

    def predict(self):

        self.load_btn.setEnabled(False)
        self.prediction_btn.setEnabled(False)
        self.prediction_cnt_btn.setEnabled(False)
        self.app_label.setText("Status : Predicting...")

        self.th_ml.mode = "predict"
        self.th_ml.start()

    def predict_cnt(self):

        self.load_btn.setEnabled(False)
        self.prediction_btn.setEnabled(False)
        self.prediction_cnt_btn.setEnabled(False)
        self.prediction_cnt_stop_btn.setEnabled(True)
        self.app_label.setText("Status : Continuously predicting...")

        self.th_ml.cont = True
        self.th_ml.mode = "predict_cnt"
        self.th_ml.start()

    def predict_cnt_stop(self):

        self.th_ml.cont = False

        self.load_btn.setEnabled(True)
        self.prediction_btn.setEnabled(True)
        self.prediction_cnt_btn.setEnabled(True)
        self.prediction_cnt_stop_btn.setEnabled(False)
        self.app_label.setText("Status : Idle")

    def selection_change(self, i):

        self.camera.changeLabel(self.d_ind[i])

    def change_capture_state(self, btn):
        #print("Change state ", btn.text())
        if btn.text() == "For Training":
            self.camera.setTrainingMode(True)
        elif btn.text() == "For Validation":
            self.camera.setTrainingMode(False)

    def clear_datasets(self):

        self.msg_box = QMessageBox()
        self.msg_box.setIcon(QMessageBox.Warning)
        self.msg_box.setText("Are you sure you want to delete all datasets?")
        self.msg_box.setInformativeText("This action cannot be undo.")
        self.msg_box.setWindowTitle("Removing Training Datasets")
        self.msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        self.msg_box.buttonClicked.connect(self.remove_datasets)

        self.msg_box.exec_()

    def remove_datasets(self, i):
        print(i.text())
        if i.text() == "&OK":
            self.data.removeDatasets()
            print("Done removing")

    def close_event(self, event):

        close = QMessageBox.question(self, "QUIT", "Sure?", QMessageBox.Yes | QMessageBox.No)

        if close == QMessageBox.Yes:
            self.th_ml.cont = False
            self.th_im.capture = False
            tm.sleep(1)
            event.accept()
        else:
            event.ignore()

    def init_ui(self):

        self.app_label = QLabel("Status : Idle")

        # Camera Preview
        cam_preview_lbl = QLabel("Camera Preview")
        self.cam_prev = QLabel()
        self.cam_prev.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.cam_prev.setFixedSize(320, 240)

        # Widget for Training Control
        select_training_lbl = QLabel("Select Label:")
        self.label_cb = QComboBox()

        for i in range(0, len(self.d_lbl)):
            self.label_cb.addItem(self.d_lbl[i])
        self.label_cb.currentIndexChanged.connect(self.selection_change)

        capture_btn = QPushButton("Capture")
        capture_btn.clicked.connect(self.capture_image)
        self.train_rb = QRadioButton("For Training")
        self.train_rb.toggled.connect(lambda: self.change_capture_state(self.train_rb))
        self.train_rb.setChecked(True)
        self.valid_rb = QRadioButton("For Validation")
        self.valid_rb.toggled.connect(lambda: self.change_capture_state(self.valid_rb))
        self.start_train_btn = QPushButton("Start Training")
        self.start_train_btn.clicked.connect(self.start_training)
        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.clicked.connect(self.clear_datasets)

        # Widget for Testing Control
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_weight)
        self.prediction_btn = QPushButton("Predict")
        self.prediction_btn.clicked.connect(self.predict)
        self.prediction_btn.setEnabled(False)
        self.prediction_cnt_btn = QPushButton("Predict Cont")
        self.prediction_cnt_btn.clicked.connect(self.predict_cnt)
        self.prediction_cnt_btn.setEnabled(False)
        self.prediction_cnt_stop_btn = QPushButton("Stop")
        self.prediction_cnt_stop_btn.clicked.connect(self.predict_cnt_stop)
        self.prediction_cnt_stop_btn.setEnabled(False)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.app_label)
        top_layout.addStretch(1)

        mid_layout = QVBoxLayout()
        mid_layout.addWidget(cam_preview_lbl)
        mid_layout.addWidget(self.cam_prev)
        mid_layout.addStretch(1)

        train_label = QHBoxLayout()
        train_label.addWidget(select_training_lbl)
        train_label.addWidget(self.label_cb)
        train_ctrl_btns = QHBoxLayout()
        train_ctrl_btns.addWidget(self.train_rb)
        train_ctrl_btns.addWidget(self.valid_rb)
        train_ctrl_btns1 = QHBoxLayout()
        train_ctrl_btns1.addWidget(self.clear_btn)
        train_ctrl_btns1.addWidget(capture_btn)
        train_ctrl_btns2 = QHBoxLayout()
        train_ctrl_btns2.addWidget(self.start_train_btn)
        train_layout = QVBoxLayout()
        train_layout.addLayout(train_label)
        train_layout.addLayout(train_ctrl_btns)
        train_layout.addLayout(train_ctrl_btns1)
        train_layout.addLayout(train_ctrl_btns2)
        train_layout.addStretch(1)
        train_ctrl_grp = QGroupBox("Training Control")
        train_ctrl_grp.setLayout(train_layout)

        test_ctrl_btns = QHBoxLayout()
        test_ctrl_btns.addWidget(self.load_btn)
        test_ctrl_btns.addWidget(self.prediction_btn)
        test_ctrl_btns1 = QHBoxLayout()
        test_ctrl_btns1.addWidget(self.prediction_cnt_btn)
        test_ctrl_btns1.addWidget(self.prediction_cnt_stop_btn)
        test_layout = QVBoxLayout()
        test_layout.addLayout(test_ctrl_btns)
        test_layout.addLayout(test_ctrl_btns1)
        test_layout.addStretch(1)
        test_ctrl_grp = QGroupBox("Testing Control")
        test_ctrl_grp.setLayout(test_layout)

        main_layout = QGridLayout()
        main_layout.addLayout(top_layout, 0, 0, 1, 2)
        main_layout.addLayout(mid_layout, 1, 0, 1, 2)
        main_layout.addWidget(train_ctrl_grp, 2, 0, 1, 2)
        main_layout.addWidget(test_ctrl_grp, 3, 0, 1, 2)
        main_layout.setRowStretch(1, 1)
        main_layout.setRowStretch(2, 1)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)

        finish = QAction("Quit", self)
        finish.triggered.connect(self.close_event)

        self.msg_box = None

        self.setLayout(main_layout)

        self.setWindowTitle("Blynk Controller Apps")

if __name__ == "__main__":

    import sys

    CAMERA = webcam.webcam()
    DATA = data.data()

    APP = QApplication([])
    MAINAPP = MainApplication(camera=CAMERA, data_mdl=DATA)
    MAINAPP.show()
    sys.exit(APP.exec_())
