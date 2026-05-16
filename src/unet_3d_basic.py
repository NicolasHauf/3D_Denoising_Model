"""
Created on Sunday Aug 03 12:37:00 2025
Expanded on 2025
Cleaned up 2026

@author: Nicolas Hauf, Dongwon Lee.
@nicolashauf@gmail.com
@dongwon.lee@mechbau.uni-stuttgart.de
@ 3D U-net model architecture
"""




from keras.optimizers import * 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import keras.layers as layers
import keras.models as models


class unet_v3():
    def __init__(self):
        print ('initiating U-net...')


    def U_net_model(self, img_shape, num_class, start_neurons):
        inputs = layers.Input(shape = img_shape)

        n = layers.BatchNormalization()(inputs)
        conv1 = layers.Conv3D(start_neurons, (3, 3, 3), activation='relu', padding='same')(n)
        conv1 = layers.Conv3D(start_neurons, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = layers.Conv3D(start_neurons*2, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv3D(start_neurons*2, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = layers.Conv3D(start_neurons*4, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv3D(start_neurons*4, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = layers.Conv3D(start_neurons*8, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv3D(start_neurons*8, (3, 3, 3), activation='relu', padding='same')(conv4)
        uconv4 = layers.UpSampling3D(size=(2, 2, 2))(conv4)
        
        cconv3 = layers.Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv3)

        up5 = layers.concatenate([uconv4, cconv3], axis=4)
        conv5 = layers.Conv3D(start_neurons*4, (3, 3, 3), activation='relu', padding='same')(up5)
        conv5 = layers.Conv3D(start_neurons*4, (3, 3, 3), activation='relu', padding='same')(conv5)

        uconv5 = layers.UpSampling3D(size=(2, 2, 2))(conv5)

        cconv2 = layers.Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv2)

        up6 = layers.concatenate([uconv5, cconv2], axis=4) 
        conv6 = layers.Conv3D(start_neurons*4, (3, 3, 3), activation='relu', padding='same')(up6)
        conv6 = layers.Conv3D(start_neurons*4, (3, 3, 3), activation='relu', padding='same')(conv6)

        uconv6 = layers.UpSampling3D(size=(2, 2, 2))(conv6)

        cconv1 = layers.Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv1)
        up7 = layers.concatenate([uconv6, cconv1], axis=4)
        conv7 = layers.Conv3D(start_neurons*2, (3, 3, 3), activation='relu', padding='same')(up7)
        conv7 = layers.Conv3D(start_neurons*2, (3, 3, 3), activation='relu', padding='same')(conv7)
        
        conv8 = layers.Conv3D(num_class, (1, 1, 1) ,activation = 'sigmoid',padding = 'same')(conv7)
        # tanh, softmax, relu, sigmoid
        
        model = models.Model(inputs=inputs, outputs=conv8)

        print(model.summary())
        return model
    