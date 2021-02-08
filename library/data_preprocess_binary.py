#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
import matplotlib.colors as mcolors


#define paths, batch size and image width and height
dir = os.getcwd()
parent_dir = os.path.dirname(dir)

PATH = parent_dir + '/data/fat'                             
PATH_IMAGES = PATH + '/images/'
PATH_MASKS = PATH + '/masks/'
PATH_PREDICTIONS = PATH + '/predictions' 

import sys
sys.path.append(parent_dir)


BATCH =4
W=256
H=256


# Takes path to images (maskes and images). 
# Define train, val and test split (test & val split are equal in this case each 0.1)

def load_data(path_images, path_masks, split=0.1):
  '''Takes path to images (maskes and images). Define train, val and test split (test & val split are equal in this case each 0.1). 
  Output is a tupel of list of strings. e.g. train_x is a list of file paths to the images selected for training'''

  images = sorted(glob(os.path.join(path_images, "*"))) #path to ultrasound images
  masks = sorted(glob(os.path.join(path_masks, "*"))) #path to masks

  total_size = len(images)
  valid_size = int(split * total_size)
  test_size = int(split * total_size)

  train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
  train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

  train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
  train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

  return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(x, width=256, height=256):
  ''' reads image as a numpy array, resizes it, and normalizes it (pixel values 0 to 1), data format float'''

  x = cv2.imread(x, cv2.IMREAD_COLOR)
  x = cv2.resize(x, (width, height))
  x = x / 255.0
  x = x.astype(np.float32)
  return x

def read_mask(x, width=256, height=256):
    ''' reads mask as a numpy array, resizes it, and normalizes it (pixel values 0 or 1), data format float'''

    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (width, height), interpolation = cv2.INTER_NEAREST)       # resize without interpolation to prevent new pixel values
    x[np.where(x == 255)] = 1                                                   # set pixel values 255 to 1. ATTENTION: this needs to be changed if masks have different pixel values 
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x
  
def tf_dataset(x, y, batch=2, width=256, height=256):
  ''' creates a tensorflow dataset'''

  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.shuffle(buffer_size=5000)
  dataset = dataset.map(lambda x, y: preprocess(x, y, width=width, height=height))
  dataset = dataset.batch(batch)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(2)
  return dataset

def preprocess(x, y, width=256, height=256):
  ''' reads images for the dataset function '''
  
  def f(x, y):
      x = x.decode()
      y = y.decode()
      image = read_image(x, width, height)
      mask = read_mask(y, width, height)
      return image, mask
  image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
  image.set_shape([height, width, 3])
  mask.set_shape([height, width, 1])
  return image, mask

def read_original_size(image_path):
  x = cv2.imread(image_path, cv2.IMREAD_COLOR)
  shape = x.shape
  width = shape[1]
  height = shape[0]
  color_channels = shape[2]
  return width, height, color_channels


if __name__ == "__main__":
     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path_images = PATH_IMAGES, path_masks = PATH_MASKS,split = 0.1)
     print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")
     train_dataset = tf_dataset(train_x, train_y, batch=BATCH, width=W, height=H)
     valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH, width=W, height=H)
     test_dataset = tf_dataset(test_x, test_y, batch=BATCH, width=W, height=H)
