#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2


from data_preprocess_binary import load_data, tf_dataset, read_image
from metrics import binary_dice_coef, binary_dice_loss

H = 256
W = 256
num_classes = 3

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)


""" Paths """
dir = os.getcwd()
parent_dir = os.path.dirname(dir)

PATH = parent_dir + '/data/fat'                             
PATH_IMAGES = PATH + '/images/'
PATH_MASKS = PATH + '/masks/'
PATH_PREDICTIONS = PATH + '/predictions' 

import sys
sys.path.append(parent_dir)


""" Dataset """
# we load data 
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path_images = PATH_IMAGES, path_masks = PATH_MASKS,split = 0.1)
print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

""" Model """
#load pretrained mobile_net or unet to predict on test data
#load mobilenet: binary_mobile_netv2.h5
#load unet: binary_unet_large_multiclass.h5

model = tf.keras.models.load_model(parent_dir + '/models/binary_mobilenet.h5',
                                            custom_objects={"binary_dice_loss": binary_dice_loss,
                                                            "binary_dice_coef": binary_dice_coef,
                                                            },compile=True)
                                            


""" Saving the masks """

model_input_data = test_x                                                       
binary_threshold = 0.5                                                          # pixel values above will be counted as 1 (fat)

for file_path in tqdm(model_input_data):
  file_name = os.path.basename(file_path)
  file_name, fileext = os.path.splitext(file_name)
  
  img = read_image(file_path, WIDTH, HEIGHT)                                    # reads image as a numpy array (IMAGE_SIZE_, IMAGE_SIZE_, 3)
  img = model.predict(np.expand_dims(img, axis=0))[0]                           # adds a dimension, makes prediction, and removes dimension again
  img[np.where(img > binary_threshold)] = 4 
  img[np.where(img <= binary_threshold)] = 0
  img = img.astype(np.uint8)
  img = np.concatenate([img, img, img], axis=2)                                 # create 3 equal channels for RGB

  orig_SIZE = read_original_size(file_path)
  img = cv2.resize(img, (orig_SIZE[0],                                          # resizes from (IMAGE_SIZE_, IMAGE_SIZE_, 3) to 
                         orig_SIZE[1]), interpolation = cv2.INTER_NEAREST)      # (IMAGE_SIZE_PLOTTING_WIDTH, IMAGE_SIZE_PLOTTING_HEIGHT)
  
 
  result_filepath = os.path.join(PATH_PREDICTIONS, 
                    "%s_prediction%s" % (file_name, fileext))
  cv2.imwrite(result_filepath, img)

