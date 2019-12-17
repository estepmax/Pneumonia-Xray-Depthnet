from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Conv2D,Input,Dense,Dropout,MaxPool2D,Flatten,BatchNormalization
from tensorflow.keras.optimizers import SGD,Adam,Nadam

from tensorflow.keras.applications import VGG16

import time 

class DepthNet(object):
    
    NAME = 'DepthNet'

    def __init__(self):
        
        self.model = None
        self.history = None
            
    def initialize(self):
        
        inputs = Input(shape=(224,224,3),name='inputs')
        
        c_64_0 = Conv2D(64,(3,3),activation='relu',padding='same',name='c_64_0')(inputs)
        c_64_1 = Conv2D(64,(3,3),activation='relu',padding='same',name='c_64_1')(c_64_0)
        mp_0 = MaxPool2D((2,2),name='mp_0')(c_64_1)
        
        c_128_0 = Conv2D(128,(3,3),activation='relu',padding='same',name='c_128_0')(mp_0)
        c_128_1 = Conv2D(128,(3,3),activation='relu',padding='same',name='c_128_1')(c_128_0)
        mp_1 = MaxPool2D((2,2),name='mp_1')(c_128_1)
        
        c_256_0 = Conv2D(256,(3,3),activation='relu',padding='same',name='c_256_0')(mp_1)
        b_0 = BatchNormalization(name='b_0')(c_256_0)
        c_256_1 = Conv2D(256,(3,3),activation='relu',padding='same',name='c_256_1')(b_0)
        b_1 = BatchNormalization(name='b_1')(c_256_1)
        c_256_2 = Conv2D(256,(3,3),activation='relu',padding='same',name='c_256_2')(b_1)
        mp_2 = MaxPool2D((2,2),name='mp_2')(c_256_2)
        
        c_512_0 = Conv2D(512,(3,3),activation='relu',padding='same',name='c_512_0')(mp_2)
        b_2 = BatchNormalization(name='b_2')(c_512_0)
        c_512_1 = Conv2D(512,(3,3),activation='relu',padding='same',name='c_512_1')(b_2)
        b_3 = BatchNormalization(name='b_3')(c_512_1)
        c_512_2 = Conv2D(512,(3,3),activation='relu',padding='same',name='c_512_2')(b_3)
        mp_3 = MaxPool2D((2,2),name='mp_3')(c_512_2)
        
        flt_0 = Flatten(name='flt_0')(mp_3)
        fc_0 = Dense(1024,activation='relu',name='fc_0')(flt_0)
        d_0 = Dropout(0.6)(fc_0)
        fc_1 = Dense(128,activation='relu',name='fc_1')(d_0)
        d_1 = Dropout(0.5)(fc_1)

        outputs = Dense(2,activation='softmax',name='output')(d_1)
        
        model = Model(inputs=inputs,outputs=outputs)
        
        optimizer = SGD(learning_rate=1e-4,clipnorm=1.0)
        #optimizer = Adam(learning_rate=1e-4,decay=1e-4)
        #optimizer = Nadam(learning_rate=1e-4)

        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=optimizer)
        
        print(model.summary())

        self.model = model

    def fit(self,generator,settings):
        
        self.history = self.model.fit_generator(generator,**settings) 
        
    def predict(self,inputs):
        return self.model.predict(inputs)
    
    #should set model weights for transfer learning as so: 
    #initialize model -> set model weights from transfer model 
    # -> freeze layers -> proceed to model training
    #TODO: create a mapping of transfer weights -> trainable weights that 
    #will take care of getting/setting weights from transfer model 
    def set_weights(self):  
        vgg16 = VGG16(weights='imagenet',include_top=False)
        
        weights,biases = vgg16.layers[1].get_weights()
        self.model.get_layer('c_64_0').set_weights([weights,biases])
        self.model.get_layer('c_64_0').trainable = False

        weights,biases = vgg16.layers[2].get_weights()
        self.model.get_layer('c_64_1').set_weights([weights,biases])
        self.model.get_layer('c_64_1').trainable = False
        
        weights,biases = vgg16.layers[4].get_weights()
        self.model.get_layer('c_128_0').set_weights([weights,biases])
        self.model.get_layer('c_128_0').trainable = False
        
        weights,biases = vgg16.layers[5].get_weights()
        self.model.get_layer('c_128_1').set_weights([weights,biases])
        self.model.get_layer('c_128_1').trainable = False
        
    def save_model(self,name):
        print('saving model weights to disk....')
        created = time.strftime("%d-%Y-%m")
        self.model.save("./depthnet/trained_models/{}-{}-{}.h5".format(DepthNet.NAME,created,name))
    
    def load_model(self,name):
        print('loading model from disk....')
        return load_model("./depthnet/trained_models/{}".format(name))

