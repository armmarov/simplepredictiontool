import cv2
import numpy as np
import datetime

class webcam:

    def __init__(self):

        self.vc = cv2.VideoCapture(0)

        self.isTrainingMode = True

        self.label_prefix = '001'

        self.img = None
        self.imgCrop = None

        self.newPict = False
    
    def changeLabel(self, label):

        print("Change label to " + label)
        
        self.label_prefix = label

        return self.label_prefix
    
    def captureImage(self, w, h):

        if self.vc.isOpened():
            rval, self.img = self.vc.read()
            self.drawContour(w, h)  
            return True
        return False        
    
    def getImage(self):
        return self.img
    
    def getCropImage(self):
        return self.imgCrop

    def saveImage(self):

        #current_dt = datetime.datetime.strptime(str(datetime.datetime.now()), '%Y-%m-%d_%H%M%S%f')
        now = datetime.datetime.now()
        if self.isTrainingMode:
            savePath = "datasets/train/" + self.label_prefix + '_' + str(datetime.datetime.timestamp(now)) + '.jpg'
        else:
            savePath = "datasets/test/" + self.label_prefix + '_' + str(datetime.datetime.timestamp(now)) + '.jpg'

        cv2.imwrite(savePath, self.imgCrop)
        return "Success", savePath
    
    def closeCamera(self):

        self.vc.release()
    
    def setTrainingMode(self, isTraining):

        self.isTrainingMode = isTraining
        #print(self.setTrainingMode)
    
    def getNewPict(self):
        return self.newPict
    
    def drawContour(self, w_resize, h_resize):

        h_org, w_org, c_org = self.img.shape

        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        img_inv = 255-img_bin

        contours, hier = cv2.findContours(img_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            approx = cv2.approxPolyDP(c, 0.07 * cv2.arcLength(c, True), True)
            
            if (len(approx) >= 4) and (w != w_org and h != h_org) and (w > w_org/3 or h > h_org/3):
                self.newPict = True
                cv2.rectangle(self.img, (x,y), (x+w, y+h), (0,0,255), 2)
                self.imgCrop = cv2.resize(self.img[y:y+h, x:x+w], (int(w_resize), int(h_resize)), interpolation=cv2.INTER_CUBIC)
                break
            else:
                self.newPict = False

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