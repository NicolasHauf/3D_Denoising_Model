# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:25:03 2020
Expanded on 2025
Cleaned up 2026

@author: Nicolas Hauf,Dongwon Lee.
@nicolashauf@gmail.com
@dongwon.lee@mechbau.uni-stuttgart.de
@ U-net training procedure
"""


# important variables are: epoch, validation_split, validation_steps, batch_size

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import unet_3d_flexible as u3f
from keras import models
from keras.optimizers import Adam
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
try:
    from keras import backend as K
except:
    from tensorflow.keras import backend as K


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)



# setting parameters and the paths for training data

tpath = os.getcwd()

# scratch folder for training data and model saving
scratch = "/scratch/users/haufns/denoising_model/"

train_path = os.path.join(scratch, 'training_data')

batch_size = int(sys.argv[4])

img_size = (64, 64, 64)


target_n = ['org_frag_data_3d']
seg_n = ['seg_frag_data_3d']


seed = 1

savetrainingpath = os.path.join(scratch + 'models')

num_of_classes = 1


# data augmentation

data_gen_args = dict(horizontal_flip = True, 
                     vertical_flip= True,
                     validation_split = 0.2,
                     brightness_range = [0.9, 1.0], 
                     fill_mode = 'nearest'
                     )


print('Training set(x): ')
x_datagen = ImageDataGenerator(**data_gen_args,
                               rescale = 1./255)

x_gen = x_datagen.flow_from_directory(
    train_path,
    classes = target_n,
    target_size = img_size,
    class_mode = None,
    color_mode = 'grayscale',
    batch_size = batch_size,
    seed = seed,
    subset = 'training'
    )

print('Training set(y): ')
y_datagen = ImageDataGenerator(**data_gen_args,
                               rescale = 1./255)

y_gen = y_datagen.flow_from_directory(
    train_path,
    classes = seg_n,
    target_size = img_size,
    class_mode = None,
    color_mode = 'grayscale',
    batch_size = batch_size,
    seed = seed,
    subset = 'training'
    )

print('Validation set(x): ')
x_val_gen = x_datagen.flow_from_directory(
    train_path,
    classes = target_n,
    target_size = img_size,
    class_mode = None,
    color_mode = 'grayscale',
    batch_size = batch_size,
    seed = seed,
    subset = 'validation'
    )

print('Validation set(y): ')
y_val_gen  = y_datagen.flow_from_directory(
    train_path,
    classes = seg_n,
    target_size = img_size,
    class_mode = None,
    color_mode = 'grayscale',
    batch_size = batch_size,
    seed = seed,
    subset = 'validation'
    )

if x_gen.n != y_gen.n or x_val_gen.n != y_val_gen.n:
    print("Number of samples in x and y generators do not match.")
    sys.exit(1)

if x_gen.n == 0 or y_gen.n == 0 or x_val_gen.n == 0 or y_val_gen.n == 0:
    print("No samples found in one or more of the generators.")
    print("Directory structure should be:\n", 
          os.path.join(train_path, target_n[0]),
          "\n", 
          os.path.join(train_path, seg_n[0]))
    sys.exit(1)

train_steps = x_gen.n // x_gen.batch_size
valid_steps = x_val_gen.n // x_val_gen.batch_size


def combine_gen(g1, g2):
    while True:
        yield (next(g1), next(g2))


train_gen = combine_gen(x_gen, y_gen)
valid_gen = combine_gen(x_val_gen,y_val_gen)




finetuning = int(sys.argv[5])
data_use = 0.5
n_epochs = 100
shuffle = 1
l_r = float(sys.argv[2])

#checkpoint_model_file = "../../newest_model.keras"
checkpoint_model_file="_"

kernel_size = int(sys.argv[3])

model_version = int(sys.argv[1])



if os.path.isfile(checkpoint_model_file):
    print("Using checkpoint")
    model = models.load_model(checkpoint_model_file)
elif model_version == 0:
    model = u3f.unet().U_net_model(img_size + tuple([1]),
                               num_class = num_of_classes,start_neurons=32, k_s=kernel_size, p_s=2)


m_name = 'model_mv{}_ml{}_is{}_e{}_ts{}_bs{}_s{}_shu{}_lr{}_ks{}_ft{}'.format(model_version, len(model.layers), img_size[0], n_epochs, int(train_steps * data_use), batch_size, seed, shuffle, l_r, kernel_size, finetuning)
savemodelname = m_name + '.keras'
savecheckname = m_name + '_checkpoint.keras'
hist_csv_file = m_name + '.csv'
plt_file = m_name + '.jpg'

print(m_name)





model.compile(optimizer = Adam(learning_rate = l_r), 
               loss = "binary_crossentropy",
               metrics = ['accuracy', 'MSE', 'MAE', 'cosine_similarity', 'binary_crossentropy'])
model_checkpoint = ModelCheckpoint(os.path.join(train_path, savecheckname),
                 monitor = 'loss',
                 verbose = 2,
                 save_best_only = True)

# history - training recording unet_v2

history = History()



def tp(model):
    return sum(np.prod(t_w.shape) for t_w in model.trainable_weights)


if finetuning == 1:
    print("Training entire model({}):".format(tp(model)))

    model_history = model.fit(train_gen,
                    batch_size=batch_size,
                    steps_per_epoch = int(train_steps * data_use),
                    epochs = int(n_epochs / 2),
                    callbacks = [model_checkpoint, history],
                    shuffle = shuffle,
                    validation_data = valid_gen,
                    validation_steps = int(valid_steps * data_use),
                    verbose = 2)

    for layer in model.layers:
        if not ("01" in layer.name or "02" in layer.name or "03" in layer.name or "07" in layer.name or "08" in layer.name or "09" in layer.name or "10" in layer.name):
            layer.trainable = False 
    print("Finetuning({}):".format(tp(model)))

    model_history = model.fit(train_gen,
                    batch_size=batch_size,
                    steps_per_epoch = int(train_steps * data_use),
                    epochs = int(n_epochs / 4),
                    callbacks = [model_checkpoint, history],
                    shuffle = shuffle,
                    validation_data = valid_gen,
                    validation_steps = int(valid_steps * data_use),
                    verbose = 2)

    for layer in model.layers:
        if not ("01" in layer.name or "09" in layer.name or "10" in layer.name):
            layer.trainable = False
        else:
            layer.trainable = True

    print("Finetuning({}):".format(tp(model)))

    model_history = model.fit(train_gen,
                    batch_size=batch_size,
                    steps_per_epoch = int(train_steps * data_use),
                    epochs = int(n_epochs / 4),
                    callbacks = [model_checkpoint, history],
                    shuffle = shuffle,
                    validation_data = valid_gen,
                    validation_steps = int(valid_steps * data_use),
                    verbose = 2)

    for layer in model.layers:
        layer.trainable = True 
elif finetuning == 2:
    print("Training entire model({}):".format(tp(model)))

    model_history = model.fit(train_gen,
                    batch_size=batch_size,
                    steps_per_epoch = int(train_steps * data_use),
                    epochs = int(n_epochs / 2),
                    callbacks = [model_checkpoint, history],
                    shuffle = shuffle,
                    validation_data = valid_gen,
                    validation_steps = int(valid_steps * data_use),
                    verbose = 2)

    for layer in model.layers:
        if not ("01" in layer.name or "02" in layer.name or "03" in layer.name or "04" in layer.name):
            layer.trainable = False 
    print("Finetuning({}):".format(tp(model)))

    model_history = model.fit(train_gen,
                    batch_size=batch_size,
                    steps_per_epoch = int(train_steps * data_use),
                    epochs = int(n_epochs / 4),
                    callbacks = [model_checkpoint, history],
                    shuffle = shuffle,
                    validation_data = valid_gen,
                    validation_steps = int(valid_steps * data_use),
                    verbose = 2)

    for layer in model.layers:
        if not ("01" in layer.name or "02" in layer.name):
            layer.trainable = False
        else:
            layer.trainable = True

    print("Finetuning({}):".format(tp(model)))

    model_history = model.fit(train_gen,
                    batch_size=batch_size,
                    steps_per_epoch = int(train_steps * data_use),
                    epochs = int(n_epochs / 4),
                    callbacks = [model_checkpoint, history],
                    shuffle = shuffle,
                    validation_data = valid_gen,
                    validation_steps = int(valid_steps * data_use),
                    verbose = 2)

    for layer in model.layers:
        layer.trainable = True 
elif finetuning == 3:
    print("Training entire model({}):".format(tp(model)))
    model_history = model.fit(train_gen,
                    batch_size=batch_size,
                    steps_per_epoch = int(train_steps * data_use),
                    epochs = int(n_epochs / 2),
                    callbacks = [model_checkpoint, history],
                    shuffle = shuffle,
                    validation_data = valid_gen,
                    validation_steps = int(valid_steps * data_use),
                    verbose = 2)
    for i in range(1, 11):
        for layer in model.layers:
            if not (f"{i:02}" in layer.name):
                layer.trainable = False 
            else:
                layer.trainable = True
        print("Finetuning({}):".format(tp(model)))

        model_history = model.fit(train_gen,
                        batch_size=batch_size,
                        steps_per_epoch = int(train_steps * data_use),
                        epochs = int(n_epochs / 20),
                        callbacks = [model_checkpoint, history],
                        shuffle = shuffle,
                        validation_data = valid_gen,
                        validation_steps = int(valid_steps * data_use),
                        verbose = 2)



    for layer in model.layers:
        layer.trainable = True 
else:
    model_history = model.fit(train_gen,
                    batch_size=batch_size,
                    steps_per_epoch = int(train_steps * data_use),
                    epochs = n_epochs,
                    callbacks = [model_checkpoint, history],
                    shuffle = shuffle,
                    validation_data = valid_gen,
                    validation_steps = int(valid_steps * data_use),
                    verbose = 2)

print("Model training completed")

model.save(os.path.join(savetrainingpath, savemodelname))


# save history
import pandas as pd

#print(model_history.history)

loss, val_loss = model_history.history['loss'], model_history.history['val_loss']
MSE, val_MSE = model_history.history['MSE'], model_history.history['val_MSE']
MAE, val_MAE = model_history.history['MAE'], model_history.history['val_MAE']
binary_crossentropy, val_binary_crossentropy = model_history.history['binary_crossentropy'], model_history.history['val_binary_crossentropy']
cosine_similarity, val_cosine_similarity = model_history.history['cosine_similarity'], model_history.history['val_cosine_similarity']


epochs = np.arange(len(loss))

i = 0
while os.path.isdir('plots/' + m_name + '_' + str(i) + '/'):
    i += 1
    
plot_folder = 'plots/' + m_name + '_' + str(i)
os.makedirs(plot_folder)

plt.figure()
plt_file = m_name + '_loss.jpg'
plt.plot(epochs, loss, label='training loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.legend()
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title(plt_file[:-4])
plt.savefig(os.path.join(plot_folder, plt_file))
plt.show()

plt.figure()
plt_file = m_name + '_MSE.jpg'
plt.plot(epochs, MSE, label='training MSE')
plt.plot(epochs, val_MSE, label='validation MSE')
plt.legend()
plt.grid()
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.title(plt_file[:-4])
plt.savefig(os.path.join(plot_folder, plt_file))
plt.show()

plt.figure()
plt_file = m_name + '_MAE.jpg'
plt.plot(epochs, MAE, label='training MAE')
plt.plot(epochs, val_MAE, label='validation MAE')
plt.legend()
plt.grid()
plt.xlabel('epochs')
plt.ylabel('MAE')
plt.title(plt_file[:-4])
plt.savefig(os.path.join(plot_folder, plt_file))
plt.show()

plt.figure()
plt_file = m_name + '_binary_crossentropy.jpg'
plt.plot(epochs, binary_crossentropy, label='training binary_crossentropy')
plt.plot(epochs, val_binary_crossentropy, label='validation binary_crossentropy')
plt.legend()
plt.grid()
plt.xlabel('epochs')
plt.ylabel('binary_crossentropy')
plt.title(plt_file[:-4])
plt.savefig(os.path.join(plot_folder, plt_file))
plt.show()

plt.figure()
plt_file = m_name + '_cosine_similarity.jpg'
plt.plot(epochs, cosine_similarity, label='training cosine_similarity')
plt.plot(epochs, val_cosine_similarity, label='validation cosine_similarity')
plt.legend()
plt.grid()
plt.xlabel('epochs')
plt.ylabel('cosine_similarity')
plt.title(plt_file[:-4])
plt.savefig(os.path.join(plot_folder, plt_file))
plt.show()


#print(model_history.history)

print("\n\n\n\n\n\n\n\n\n\n")


hist_df = pd.DataFrame(history.history) 

with open(os.path.join(savetrainingpath, hist_csv_file), mode='w') as f:
    hist_df.to_csv(f)

    
#%% to check the model size

def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
