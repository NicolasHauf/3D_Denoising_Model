# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:25:03 2020

@author: Dongwon Lee.
@dongwon.lee@mechbau.uni-stuttgart.de
@ Post-processing after U-net
"""


from img_seg_3d import *
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
print("Num GPUs Available: ", 
      len(tf.config.experimental.list_physical_devices('GPU')))


gpus  = tf.config.experimental.list_physical_devices('GPU')


if gpus:
  # Restrict TensorFlow to only use the first GPUq 
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
    

# path setting for prediction tiles

model_name = "model_mv0_ml37_is64_e100_ts1502_bs4_s1_shu1_lr0.0001_ks4"

model_v = model_name + '.keras'

frag_save_path = 'frag_results_3d'
img_size = (64, 64, 64)
num_of_classes = 1

tpath = os.path.join(os.getcwd(), '..') 
test_n = ['org_frag_data_3d']

test_data_path = os.path.join(tpath,'sample_data', 'U_net', 'training_data')


#%%
#model = load_model(os.path.join(tpath, 'results', 'U_net', model_v))
#%% 
model = load_model(os.path.join(tpath, 'results', 'U_net', model_v))

print('Model has been read!')

#%%
test_datagen = ImageDataGenerator(rescale = 1./255)
test_gen = test_datagen.flow_from_directory(
    test_data_path,  
    classes = test_n,
    color_mode = 'grayscale',
    shuffle = False,
    class_mode = None,
    batch_size = 1,
    target_size = img_size
)


# creating predictions

frlist,frpath = readingfiles(test_data_path, 'org_frag_data_3d')
frlist.sort()

t_l = len(test_gen)

a = 0
counter = 0
for i in range(int(t_l * 0.8), t_l):
    #if np.mean(test_gen[i]) != 0:
    #    a += 1
    #if i <= 100:
    #  print(test_gen[i].shape, np.mean(test_gen[i]))
    result = model.predict(test_gen[i], verbose=0)
    #break
    #btemp = io.imread(os.path.join(frpath, frlist[i]))
    atemp = result.squeeze()
    atemp[atemp<0] =0
    if i % int(len(test_gen) / 100) == 0:
        print(i / len(test_gen) * 5)
    
    saveCheckImages(atemp,
                    os.path.join(tpath,'results','U_net',frag_save_path),
                    'frag_seg_3d_'+frlist[i])
print('model prediction done!')
#%% path setting for merged files
mfpath = os.path.join(tpath,'results','U_net',frag_save_path)

mflist = os.listdir(mfpath)

merge_save_path = os.path.join('results', 'U_net', 'seg_merged_3d')

filecounter = 0
msavepath = os.path.join(tpath,merge_save_path) 
#%%      
if not os.path.exists(msavepath):
    os.makedirs(msavepath)

filecounter = 0
for iTemp in tqdm(range(int(len(mflist) / 81))):
    #mimg,filecounter = overlap_merge_image((2940,2940),filecounter,img_size,mfpath,mflist)
    mimg,filecounter = overlap_merge_image((2940,2940),(400*9,400*9),
                                           filecounter,400,256,mfpath,mflist)
    slice_number = mflist[filecounter-1].split('_')[3]
    savename = os.path.join(msavepath, "uint8_" + slice_number + ".tif")
    zeroTemp = np.zeros((2940,2940))
    shift_factor = int((400-256)/2/2 + 1)
    zeroTemp[0:2940-shift_factor,0:2940-shift_factor] = mimg[shift_factor:,shift_factor:]
    io.imsave(savename,zeroTemp.astype('uint8'))  





