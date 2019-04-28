import tensorflow as tf
import numpy as np
import os

class model:

    def __init__(self):

        self.seqmodel = None
        self.ckpt_path = "model/cp.ckpt"
    
    def createModel(self, input_shape, num_classes):

        
        self.seqmodel = tf.keras.applications.MobileNetV2(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg',
                    input_shape=input_shape
        )
        dropout = tf.keras.layers.Dropout(rate=0.25)(self.seqmodel.layers[-1].output)
        logits = tf.keras.layers.Dense(units=num_classes, activation='softmax')(dropout)
        self.seqmodel = tf.keras.models.Model(self.seqmodel.inputs, logits)
        self.seqmodel.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
        
        self.seqmodel.summary()

        return self.seqmodel
    
    def training(self, x, y, epochs, steps_per_epoch, batch):

        cp_cb = tf.keras.callbacks.ModelCheckpoint(self.ckpt_path,
                                                    save_weights_only=True,
                                                    verbose=1)

        self.seqmodel.fit(x, y, batch_size=batch, 
                            epochs=epochs, 
                            steps_per_epoch=steps_per_epoch,
                            callbacks = [cp_cb])
    
    def load_weight(self):

        self.seqmodel.load_weights(self.ckpt_path)
    
    def evaluation(self, x, y):

        self.seqmodel.evaluate(x, y)

    def predict(self, x):

        pred_res = self.seqmodel.predict(x)
        ret = np.argmax(pred_res)
        print("Prediction: ", ret, pred_res)

        return ret


