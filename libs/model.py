import tensorflow as tf
import numpy as np
import os

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
           
    def createModel(self, input_shape, num_classes):

        self.seqmodel = None
        self.weight = None
        self.ckpt_path = "model/cp.ckpt"

        self.mlgraph = tf.Graph()
        
        with self.mlgraph.as_default():
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
            print(self.mlgraph, self.seqmodel)

        return self.seqmodel
    
    def training(self, x, y, x_valid, y_valid, epochs, steps_per_epoch, batch):

        print(self.mlgraph, self.seqmodel)
        with self.mlgraph.as_default():
            cp_cb = tf.keras.callbacks.ModelCheckpoint(self.ckpt_path,
                                                        save_weights_only=True,
                                                        verbose=1)

            self.seqmodel.fit(x, y, batch_size=batch, 
                                epochs=epochs, 
                                validation_data=(x_valid, y_valid),
                                steps_per_epoch=steps_per_epoch,
                                callbacks = [cp_cb])
    
    def load_weight(self):
        #print(self.mlgraph, self.seqmodel)
        with self.mlgraph.as_default():
            self.seqmodel.load_weights(self.ckpt_path)
            #print(self.seqmodel.get_weights())
            self.weight = self.seqmodel.get_weights()
            #print(self.weight[len(self.weight) - 1])
        return self.weight
    
    def evaluation(self, x, y):

        self.seqmodel.evaluate(x, y)

    def predict(self, x):

        #print(self.mlgraph, selfa.seqmodel)
        with self.mlgraph.as_default():
            #print(self.weight[len(self.weight) - 1])
            self.seqmodel.set_weights(self.weight)
            pred_res = self.seqmodel.predict(x)
            ret = 0
            for x in pred_res[0]:
                if x > 0.90:
                    ret = np.argmax(pred_res)
                    print("Prediction: ", ret, pred_res)
    
        return ret


