"""
Created on Wed Jun 10 16:25:03 2020

@author: Nicolas Hauf, Dongwon Lee.
@nicolashauf@gmail.com
@dongwon.lee@mechbau.uni-stuttgart.de
@ 2D U-net model architecture
"""



from keras.optimizers import * 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import keras.layers as layers
import keras.models as models

class unet():
    def __init__(self):
        print ('initiating U-net...')


    def U_net_model(self, img_shape, num_class,start_neurons):

        
        inputs = layers.Input(shape = img_shape)
        n = layers.BatchNormalization()(inputs)
        conv1 = layers.Conv2D(start_neurons, (3, 3), activation='relu', padding='same')(n)
        conv1 = layers.Conv2D(start_neurons, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(start_neurons*2, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(start_neurons*2, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(start_neurons*4, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(start_neurons*4, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(start_neurons*8, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(start_neurons*8, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(start_neurons*16, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(start_neurons*16, (3, 3), activation='relu', padding='same')(conv5)
        
        uconv5 = layers.UpSampling2D(size=(2, 2))(conv5)
    
        cconv4 = layers.Cropping2D(cropping=((0,0),(0,0)))(conv4)


        up6 = layers.concatenate([uconv5, cconv4], axis=3) ######
        conv6 = layers.Conv2D(start_neurons*8, (3, 3), activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(start_neurons*8, (3, 3), activation='relu', padding='same')(conv6)

        uconv6 = layers.UpSampling2D(size=(2, 2))(conv6)
        
        cconv3 = layers.Cropping2D(cropping=((0,0),(0,0)))(conv3)

        up7 = layers.concatenate([uconv6, cconv3], axis=3) 
        conv7 = layers.Conv2D(start_neurons*4, (3, 3), activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(start_neurons*4, (3, 3), activation='relu', padding='same')(conv7)

        uconv7 = layers.UpSampling2D(size=(2, 2))(conv7)
        
        cconv2 = layers.Cropping2D(cropping=((0,0),(0,0)))(conv2)
        up8 = layers.concatenate([uconv7, cconv2], axis=3)
        conv8 = layers.Conv2D(start_neurons*2, (3, 3), activation='relu', padding='same')(up8)
        conv8 = layers.Conv2D(start_neurons*2, (3, 3), activation='relu', padding='same')(conv8)

        uconv8 = layers.UpSampling2D(size=(2, 2))(conv8)

        cconv1 = layers.Cropping2D(cropping=((0,0),(0,0)))(conv1)
        up9 = layers.concatenate([uconv8, cconv1], axis=3)
        conv9 = layers.Conv2D(start_neurons, (3, 3), activation='relu', padding='same')(up9)
        conv9 = layers.Conv2D(start_neurons, (3, 3), activation='relu', padding='same')(conv9)

        
        conv10 = layers.Conv2D(num_class, (1 , 1) ,activation = 'sigmoid',padding = 'same')(conv9)
        # tanh, softmax, relu, sigmoid
        
        model = models.Model(inputs=inputs, outputs=conv10)
        
        print(model.summary())
        return model

unet().U_net_model((400, 400) + tuple([1]), num_class = 1,start_neurons=32)

