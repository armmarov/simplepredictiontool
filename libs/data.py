import glob
import os
import cv2
from xml.dom import minidom
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class data:

    def __init__(self):

        self.parent_dir = "./datasets"
        self.train_dir = self.parent_dir + "/train"
        self.test_dir = self.parent_dir + "/test"
    
    def __loadImages(self, path):

        data = []
        label = []
        files = []

        for ext in ('*.jpg', '*.png', '*.jpeg', '*.gif', '*.bmp'):
            files.extend(glob.glob(os.path.join(path, ext)))
        
        for file in files:
            bgr_img = cv2.imread(file)
            data.append(self.resize(bgr_img))
            filename = os.path.basename(file)
            label.append(int(filename[0:3]))

        return data, label

    def genImage(self):
        
        np.random.seed(1)
        seed=1337

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                    target_size=(self.height,self.width),
                                                    batch_size=10,
                                                    seed=seed,
                                                    shuffle=True,
                                                    class_mode='categorical')

        # Test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = test_datagen.flow_from_directory(self.test_dir,
                                                  target_size=(self.height,self.width),
                                                  batch_size=10,
                                                  seed=seed,
                                                  shuffle=False,
                                                  class_mode='categorical')

        return train_generator, validation_generator
        

    def setImgSize(self, width, height, channel):

        self.width = width
        self.height = height
        self.channel = channel
    
    def loadData(self, isTraining):

        if isTraining:
            return self.__loadImages(self.train_dir + "/*/")
        else:
            return self.__loadImages(self.test_dir + "/*/")
    
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
