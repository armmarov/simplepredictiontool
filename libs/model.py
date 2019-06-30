import tensorflow as tf
import numpy as np
import os
import cv2
import random

class model:

    class __model:
        def __init__(self):
            self.val = None

        def __str__(self):
            return self.val
    
    instance = None

    def __init__(self):

        if not model.instance:
            model.instance = model.__model
           
    def createModel(self, input_shape, num_classes, model_type="ud1"):

        self.seqmodel = None
        self.weight = None
        self.ch = input_shape[2]
        self.ckpt_path = "model/cp.ckpt"

        self.mlgraph = tf.Graph()
        
        with self.mlgraph.as_default():
            
            if model_type == "mobilenetv2":
                self.seqmodel = self.model_mobilenetv2(input_shape, num_classes)
            elif model_type == "ud1":
                self.seqmodel = self.model_userdefined1(input_shape, num_classes)

            self.seqmodel.summary()
            print(self.mlgraph, self.seqmodel)

        return self.seqmodel
    
    def training(self, x, y, x_valid, y_valid, **kwargs):
        
        epochs = kwargs['epochs']
        steps_per_epoch = kwargs['steps_per_epoch']
        batch = kwargs['batch']
        optimizer = kwargs['optimizer']
        loss = kwargs['loss']

        print(self.mlgraph, self.seqmodel)

        with self.mlgraph.as_default():

            self.seqmodel.compile(optimizer=optimizer,
                                loss=loss,
                                metrics=['accuracy'])
            
            cp_cb = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(self.ckpt_path, save_weights_only=True, verbose=1, save_best_only=True)
            ]

            tf.keras.initializers.glorot_normal(seed=None) # Initialize using xavier
            tf.keras.backend.get_session().run(tf.global_variables_initializer())
               
            x_pp_org = []
            for i in range (0, len(x)):
                if self.ch == 1:
                    x_pp_org.append(self.preprocessing_binary(x[i], y[i]))
                else:
                    x_pp_org.append(self.preprocessing_normalize(x[i], y[i]))
            x_pp = np.array(x_pp_org)
            x_pp = np.reshape(x_pp, (len(x_pp_org), len(x_pp_org[0]), len(x_pp_org[1]), self.ch ))

            x_valid_pp_org = []
            for j in range(0, len(x_valid)):
                if self.ch == 1:
                    x_valid_pp_org.append(self.preprocessing_binary(x_valid[j], y_valid[j]))
                else:
                    x_valid_pp_org.append(self.preprocessing_normalize(x_valid[j], y_valid[j]))
            x_valid_pp = np.array(x_valid_pp_org)
            x_valid_pp = np.reshape(x_valid_pp, (len(x_valid_pp_org),len(x_valid_pp_org[0]), len(x_valid_pp_org[1]), self.ch ))

            self.seqmodel.fit(x_pp, y, batch_size=batch, 
                                epochs=epochs, 
                                validation_data=(x_valid_pp, y_valid),
                                steps_per_epoch=steps_per_epoch,
                                callbacks = cp_cb)
            self.weight = self.seqmodel.get_weights()
    
    def load_weight(self):
        #print(self.mlgraph, self.seqmodel)
        with self.mlgraph.as_default():
            self.seqmodel.load_weights(self.ckpt_path)
            self.weight = self.seqmodel.get_weights()
        return self.weight
    
    def evaluation(self, x, y):

        self.seqmodel.evaluate(x, y)

    def predict(self, x):

        #print(self.mlgraph, selfa.seqmodel)
        with self.mlgraph.as_default():
            #print(self.weight[len(self.weight) - 1])
            self.seqmodel.set_weights(self.weight)
            
            if self.ch == 1:
                x_pp_org = self.preprocessing_binary(x)
            else:
                x_pp_org = self.preprocessing_normalize(x)

            x_pp = np.array(x_pp_org)
            x_pp = np.reshape(x_pp, (1, len(x_pp_org[0]), len(x_pp_org[1]), self.ch ))
            
            pred_res = self.seqmodel.predict(x_pp)
            ret = 0
            for _p in pred_res[0]:
                if _p > 0.90:
                    ret = np.argmax(pred_res)
                    print("Prediction: ", ret, pred_res)
    
        return ret

    def model_mobilenetv2(self, _input, _classes):

        base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=_input
                )

        fc_1 = tf.keras.layers.Dense(512, activation='relu')(base_model.layers[-1].output)
        fc_1 = tf.keras.layers.Dropout(0.75)(fc_1)
        fc_2 = tf.keras.layers.Dense(128, activation='relu')(fc_1)
        fc_2 = tf.keras.layers.Dropout(0.5)(fc_2)

        logits = tf.keras.layers.Dense(units=_classes, activation='softmax')(fc_2)

        for lyr in base_model.layers:
            print("freeze", lyr)
            lyr.trainable=False

        return tf.keras.models.Model(base_model.inputs, logits)

    def model_userdefined1(self, _input, _classes):

        input_layer = tf.keras.Input(shape=_input, name='input_layer')

        # Some convolutional layers
        conv_1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
        conv_1 = tf.keras.layers.MaxPooling2D(padding='same')(conv_1)
        conv_2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(conv_1)
        conv_2 = tf.keras.layers.MaxPooling2D(padding='same')(conv_2)

        # Flatten the output of the convolutional layers
        conv_flat = tf.keras.layers.Flatten()(conv_2)

        # Some dense layers with two separate outputs
        fc_1 = tf.keras.layers.Dense(128, activation='relu')(conv_flat)
        fc_1 = tf.keras.layers.Dropout(0.2)(fc_1)
        fc_2 = tf.keras.layers.Dense(128, activation='relu')(fc_1)
        fc_2 = tf.keras.layers.Dropout(0.2)(fc_2)

        output_layer = tf.keras.layers.Dense(_classes, activation='softmax', name='logits')(fc_2)

        return tf.keras.models.Model(input_layer, output_layer)
    
    def preprocessing_binary(self, img, lbl=None):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        img_inv = 255-img_bin

        if lbl != None:
            cv2.imwrite("datasets/temp/res_" + str(random.randint(1,200)) + "_" + str(lbl) + ".jpg", img_inv)
        
        return img_inv
    
    def preprocessing_combine(self, img, lbl=None):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        img_inv = 255-img_bin

        if lbl != None:
            cv2.imwrite("datasets/temp/res_" + str(random.randint(1,200)) + "_" + str(lbl) + ".jpg", img_inv)
        
        return img_inv
    
    def preprocessing_normalize(self, img, lbl=None):

        h_org, w_org, c_org = img.shape

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        if lbl != None:
            cv2.imwrite("datasets/temp/res_" + str(random.randint(1,200)) + "_" + str(lbl) + ".jpg", img_output)
        
        return img_output
