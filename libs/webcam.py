import cv2
import numpy as np
import datetime

class webcam:

    def __init__(self):

        self.vc = cv2.VideoCapture(0)

        self.isTrainingMode = True

        self.label_prefix = '001'
    
    def changeLabel(self, label):

        print("Change label to " + label)
        
        self.label_prefix = label

        return self.label_prefix
    
    def captureImage(self):

        if self.vc.isOpened():
            rval, frame = self.vc.read()
        else:
            rval = False
        
        if rval:
            return rval, frame
        else:
            return rval, []
    
    def saveImage(self):

        #current_dt = datetime.datetime.strptime(str(datetime.datetime.now()), '%Y-%m-%d_%H%M%S%f')
        now = datetime.datetime.now()
        if self.isTrainingMode:
            savePath = "datasets/train/" + self.label_prefix + '_' + str(datetime.datetime.timestamp(now)) + '.jpg'
        else:
            savePath = "datasets/test/" + self.label_prefix + '_' + str(datetime.datetime.timestamp(now)) + '.jpg'
        rval, frame = self.captureImage()
        if rval:
            cv2.imwrite(savePath, frame)
            return "Success", savePath
        
        return "Failed"
    
    def closeCamera(self):

        self.vc.release()
    
    def setTrainingMode(self, isTraining):

        self.isTrainingMode = isTraining

    def streamImage(self, cb):

        #cv2.namedWindow("Preview")

        if self.vc.isOpened():
            rval, frame = self.vc.read()
        else:
            rval = False
        
        while rval:
            rval, frame = self.vc.read()

            #cv2.imshow("Preview", self.__drawRect(frame))
            
            #cb(frame)

            key = cv2.waitKey(20)

            if key == 27:
                break
            elif key == 32:
                if self.isTrainingMode:
                    savePath = "datasets/train/" + self.label_prefix + '_{:>03}.jpg'
                else:
                    savePath = "datasets/test/" + self.label_prefix + '_{:>03}.jpg'
                cv2.imwrite(savePath, frame[90:450,0:640])
                print("Image is saved into ", savePath)
        
        #cv2.destroyWindow("Preview")
        self.vc.release
        return rval

    def streamImageForPredict(self, predictFunc, resizeFunc, w, h):

        cv2.namedWindow("Preview")

        if self.vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        
        while rval:
            rval, frame = self.vc.read()

            cv2.imshow("Preview", self.__drawRect(frame))
            key = cv2.waitKey(20)

            if key == 27:
                break
            elif key == 32:
                dat = []
                dat.append(resizeFunc(frame[90:450,0:640], w, h))
                cv2.imwrite("./datasets/temp/tempImg.jpg", resizeFunc(frame[90:450,0:640], w, h))
                dat_rs = np.array(dat)
                print("dat_rs", dat_rs.shape)
                predictFunc(dat_rs)
        
        cv2.destroyWindow("Preview")
        self.vc.release
        return rval

    def __drawRect(self, frame):
        cv2.rectangle(frame, (0,90), (640,450), (255,0,0), 2)
        return frame