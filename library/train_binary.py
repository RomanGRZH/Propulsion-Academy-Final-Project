#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Recall, Precision

from data_augmentation import load_augmented_x_train_y_train
from metrics import binary_dice_coef, binary_dice_loss
from data_preprocess_binary import load_data, tf_dataset
from models import binary_mobile_netv2, binary_unet_large

if __name__ == "__main__":
    
    """ Seeding """
np.random.seed(42)
tf.random.set_seed(42)
    
    
""" Loading original images and masks """

dir = os.getcwd()
parent_dir = os.path.dirname(dir)

PATH = parent_dir + '/data/fat'                             
PATH_IMAGES = PATH + '/images/'
PATH_MASKS = PATH + '/masks/'
PATH_PREDICTIONS = PATH + '/predictions' 

import sys
sys.path.append(parent_dir)

# we load data 
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path_images = PATH_IMAGES, path_masks = PATH_MASKS,split = 0.1)
print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")


""" Hyperparameters """
WIDTH = 256
HEIGHT = 256
shape = (WIDTH, HEIGHT, 3)
lr = 1e-4
EPOCHS = 50
BATCH_SIZE = 4
    
""" Model """
METRICS= [ binary_dice_coef, Recall(), Precision()]   
        
      
model = binary_mobile_netv2(shape)
model.compile(loss=binary_dice_loss, optimizer=tf.keras.optimizers.Adam(lr), metrics=METRICS)

train_dataset = tf_dataset(train_x, train_y, batch=BATCH_SIZE)
valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH_SIZE)

train_steps = len(train_x)//BATCH_SIZE
valid_steps = len(valid_x)//BATCH_SIZE
if len(train_x) % BATCH_SIZE != 0:
    train_steps += 1
if len(valid_x) % BATCH_SIZE != 0:
    valid_steps += 1

callbacks = [
        ModelCheckpoint(parent_dir + '/models/binary.h5' ,  verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=10, verbose=1)
        ]

history = model.fit(train_dataset,
        steps_per_epoch=train_steps,
        validation_data=valid_dataset,
        validation_steps=valid_steps,
        epochs=EPOCHS,
        callbacks=callbacks
    )


