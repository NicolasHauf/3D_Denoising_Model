"""
Created on Wed Jun 10 16:25:03 2020
Expanded on 2025
Cleaned up 2026

@author: Nicolas Hauf, Dongwon Lee.
@nicolashauf@gmail.com
@dongwon.lee@mechbau.uni-stuttgart.de
@ 3D customizable U-net model architecture
"""

import keras.layers as layers
import keras.models as models

class unet():
    def __init__(self):
        print ('initiating U-net...')


    def U_net_model(self, img_shape, num_class, start_neurons, k_s=3, p_s=2):

        
        inputs = layers.Input(shape = img_shape)

        n = layers.BatchNormalization()(inputs)
        conv1 = layers.Conv3D(start_neurons, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_01_a")(n)
        conv1 = layers.Conv3D(start_neurons, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_01_b")(conv1)
        #att1 = layers.Attention()(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(p_s, p_s, p_s), name="pool_01")(conv1)

        conv2 = layers.Conv3D(start_neurons*2, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_02_a")(pool1)
        conv2 = layers.Conv3D(start_neurons*2, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_02_b")(conv2)
        #att2 = layers.Attention()(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(p_s, p_s, p_s), name="pool_02")(conv2)

        conv3 = layers.Conv3D(start_neurons*4, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_03_a")(pool2)
        conv3 = layers.Conv3D(start_neurons*4, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_03_b")(conv3)
        #att3 = layers.Attention()(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(p_s, p_s, p_s), name="pool_03")(conv3)

        conv4 = layers.Conv3D(start_neurons*8, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_04_a")(pool3)
        conv4 = layers.Conv3D(start_neurons*8, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_04_b")(conv4)
        #att4 = layers.Attention()(conv4)
        pool4 = layers.MaxPooling3D(pool_size=(p_s, p_s, p_s), name="pool_04")(conv4)

        conv5 = layers.Conv3D(start_neurons*16, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_05_a")(pool4)
        conv5 = layers.Conv3D(start_neurons*16, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_05_b")(conv5)
        #att5 = layers.Attention()(conv5)

        uconv5 = layers.UpSampling3D(size=(p_s, p_s, p_s), name="up_05")(conv5)

        cconv4 = layers.Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv4)

        up6 = layers.concatenate([uconv5, cconv4], axis=4)
        conv6 = layers.Conv3D(start_neurons*8, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_06_a")(up6)
        conv6 = layers.Conv3D(start_neurons*8, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_06_b")(conv6)
        #att6 = layers.Attention()(conv6)

        uconv6 = layers.UpSampling3D(size=(p_s, p_s, p_s), name="up_06")(conv6)

        cconv3 = layers.Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv3)

        up7 = layers.concatenate([uconv6, cconv3], axis=4) 
        conv7 = layers.Conv3D(start_neurons*4, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_07_a")(up7)
        conv7 = layers.Conv3D(start_neurons*4, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_07_b")(conv7)
        #att7 = keras.layers.Attention()(conv7)

        uconv7 = layers.UpSampling3D(size=(p_s, p_s, p_s), name="up_07")(conv7)

        cconv2 = layers.Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv2)
        up8 = layers.concatenate([uconv7, cconv2], axis=4)
        conv8 = layers.Conv3D(start_neurons*2, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_08_a")(up8)
        conv8 = layers.Conv3D(start_neurons*2, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_08_b")(conv8)
        #att8 = layers.Attention()(conv8)

        uconv8 = layers.UpSampling3D(size=(p_s, p_s, p_s), name="up_08")(conv8)

        cconv1 = layers.Cropping3D(cropping=((0, 0), (0, 0), (0, 0)))(conv1)
        up9 = layers.concatenate([uconv8, cconv1], axis=4)
        conv9 = layers.Conv3D(start_neurons, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_09_a")(up9)
        conv9 = layers.Conv3D(start_neurons, (k_s, k_s, k_s), activation='relu', padding='same', name="conv_09_b")(conv9)
        #att9 = layers.Attention()(conv9)

        
        conv10 = layers.Conv3D(num_class, (1, 1, 1) ,activation = 'sigmoid',padding = 'same', name="conv_10")(conv9)
        # tanh, softmax, relu, sigmoid
        
        model = models.Model(inputs=inputs, outputs=conv10)

        print(model.summary())
        return model
    
# test instance
_ = unet().U_net_model((64, 64, 64) + tuple([1]), 1, 32, 3, 2)
