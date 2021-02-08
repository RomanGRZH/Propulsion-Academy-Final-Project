#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.colors as mcolors


def read_image_PLOTTING(path, width, height):
  ''' reads an image and resizes but WITHOUT normalizing'''
  x = cv2.imread(path, cv2.IMREAD_COLOR)
  x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)
  x = cv2.resize(x, (width, height))
  return x

def create_overlay(img, width=256, height=256, color1='red', color2='green'):
  '''createas a RGBA image in form of a numpy array. Loads binary, grayscale 
  or color image and transforms black values to color, and white values to trasparent'''
  img = img.resize((width, height), resample = Image.NEAREST)
  img= np.array(img.convert('RGBA')).astype(np.uint8) 
  mask_fat = (img[:,:,2] == 4)  
  mask_muscle = (img[:,:,2] == 1)
  img[:,:,3] = 0
  img[:,:,3][np.where(mask_fat| mask_muscle) ] = 255

  R, G, B = np.multiply(mcolors.to_rgb(color1),255).astype(np.uint8)
  img[:,:,0][np.where(mask_fat)] =R
  img[:,:,1][np.where(mask_fat)] =G
  img[:,:,2][np.where(mask_fat)] =B

  R, G, B = np.multiply(mcolors.to_rgb(color2),255).astype(np.uint8)
  img[:,:,0][np.where(mask_muscle)] =R
  img[:,:,1][np.where(mask_muscle)] =G
  img[:,:,2][np.where(mask_muscle)] =B
  return img


def crop_image(image_array, box):
  ''' crops an image to size of box'''
  cropped_image = Image.fromarray(image_array)
  cropped_image = cropped_image.crop(box) 
  return cropped_image

def convert_mask_to_area(path):
  img = Image.open(path)
  img = np.array(img)
  img[np.where(img== 255)] = 4
  img = Image.fromarray(img)                                            
  return img

def convert_mask_to_outline(path, contour_width=6):
  img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
  ret, thresh = cv2.threshold(img, 127, 255, 0)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  img_contours = np.zeros(img.shape)
  cv2.drawContours(img_contours, contours, -1, (4,4,4), contour_width)
  img = Image.fromarray(img_contours)                                            
  return img


def read_original_size(image_path):
  x = cv2.imread(image_path, cv2.IMREAD_COLOR)
  shape = x.shape
  width = shape[1]
  height = shape[0]
  color_channels = shape[2]
  return width, height, color_channels