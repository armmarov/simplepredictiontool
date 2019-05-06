import glob
import os
import cv2
from xml.dom import minidom
import numpy as np
import tensorflow as tf
import random

class data:

    def __init__(self):

        self.parent_dir = "./datasets"
        self.train_dir = self.parent_dir + "/train/"
        self.test_dir = self.parent_dir + "/test/"
    
    def __loadImages(self, path):

        data = []
        label = []

        for file in glob.glob(path + "*.jpg"):
            bgr_img = cv2.imread(file)
            data.append(self.resize(bgr_img))
            filename = os.path.basename(file)
            label.append(int(filename[0:3]))

        return data, label

    def setImgSize(self, width, height, channel):

        self.width = width
        self.height = height
        self.channel = channel
    
    def loadData(self, isTraining):

        if isTraining:
            return self.__loadImages(self.train_dir)
        else:
            return self.__loadImages(self.test_dir)
    
    def importXML(self):

        doc = minidom.parse("libs/label.xml")
        items = doc.getElementsByTagName('item')

        labels = []
        index = []
        api = []

        for elem in items:
            labels.append(elem.childNodes[1].firstChild.data)
            index.append(elem.childNodes[3].firstChild.data)
            api.append(elem.childNodes[5].firstChild.data)

        return labels, index, api

    def removeDatasets(self):

        train_files = glob.glob('./datasets/train/*.jpg')
        test_files = glob.glob('./datasets/test/*.jpg')

        for f in train_files:
            os.remove(f)
        
        for f in test_files:
            os.remove(f)

    def reformat(self, dat, label):

        datasets = dat.reshape((-1, self.height, self.width, self.channel)).astype(np.float32)
        labels = (np.arange(3) == label[:, None]).astype(np.float32)

        return datasets, labels

    def resize(self, dat):

        return cv2.resize(dat, dsize=(self.width, self.height), interpolation=cv2.INTER_CUBIC)
    
    def randomize(self, a, b):

        c = list(zip(a,b))
        random.shuffle(c)
        a,b = zip(*c)
        ret_a = np.array(a)
        ret_b = np.array(b)

        return ret_a, ret_b