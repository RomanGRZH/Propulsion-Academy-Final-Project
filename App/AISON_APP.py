"""
Created on Wed Nov 13 18:29:03 2020

@author: Leticia Fernandez Moguel

Streamlit aplication to predict the muscle and fat layer tissue from an ultrasound

"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import sample 
from PIL import Image
from datetime import datetime

dir = os.getcwd()
parent_dir = os.path.dirname(dir)
import sys
sys.path.append(parent_dir)


# MobileNet imports
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

# U-NET imports
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import time
from library.metrics import *
from library.helper import *

# The functions used in this script are contained in library folder 
# The helper library contains all the paths and functions used in the app
# the metrics


################################### The app starts ####################################
st.title('Aison Technologies (Former Caressoma)')

# Select DEMO or PREDICTION MODE as wel as MODELS
st.sidebar.title("Select Parameters")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Demo Mode", "Animation", "Prediction mode"])

if app_mode == "Animation":
    st.empty()
    calculation_mode = None
    model_type = "MobileNetV2"
    model_name = "MobileNetV2_c"
    Dataset = "fat_muscle"   
else:
    st.sidebar.success('To continue select Model Parameters.')
    calculation_mode = st.sidebar.selectbox("Tissue to predict",
        ["Fat", "Fat and muscle"])
    model_type = st.sidebar.selectbox("Choose the model",
    ["UNET", "MobileNetV2"])


# Here we have to define metrics per type of model
if calculation_mode == "Fat":  
    coeff_name = 'binary_dice_coef'
    custom_metrics = {'binary_dice_coef':binary_dice_coef, 
        'binary_dice_loss': binary_dice_loss,
        'dice_loss': dice_loss} 
    base_file = "00010.png"
    model_name = model_type + "_b"
    Dataset = "fat"
elif calculation_mode == "Fat and muscle":
    coeff_name = 'dice_coef'
    if model_type == "UNET":
        custom_metrics = {'generalized_dice_coef':generalized_dice_coef, 
            'dice_coef': dice_coef,
            'dice_coef_c1': dice_coef_c1,            
            'dice_coef_c2': dice_coef_c2, 
            'dice_coef_c3': dice_coef_c3, 
            'generalized_dice_loss': generalized_dice_loss, 
            'dice_loss': dice_loss}
    elif model_type == "MobileNetV2":
        custom_metrics = {'generalized_dice_coef':generalized_dice_coef, 
            'dice_coef': dice_coef,
            'dice_coef_c1': dice_coef_c1,            
            'dice_coef_c2': dice_coef_c2, 
            'dice_coef_c3': dice_coef_c3, 
            'generalized_dice_loss': generalized_dice_loss, 
            'dice_loss': dice_loss}
# this is for now to be changed later on    
    base_file = "00008.png"
    model_name = model_type + "_c" 
    Dataset = "fat_muscle"
path_model = Directory + "/models/" + model_name +"/"
PATH_Data = Directory + "/data/" + Dataset + "/"


if app_mode == "Demo Mode":  # A preloaded set of results
    st.header('Demo Mode')
    st.subheader(calculation_mode + " prediction with " + model_type + " model")
    if st.checkbox('Show Training History'):
        st.subheader('Training history')
        Full_history = load_history(path_model, SAVE_FORMAT)
        fig_1, axs = training_history_plot(Full_history, coeff_name)
        st.pyplot(fig_1)
    if st.checkbox('Show Results'):
        st.subheader('Predictions')
        path_pred = Get_prediction_path(path_model = path_model, base_file = base_file, epoch = 0,
                    Animation = False)
        if calculation_mode == "Fat":
            fig_2, fig_3, axs2, axs3 = predictions_plots_binary(base_file = base_file, PATH_Data = PATH_Data, prediction_image_path = path_pred)
            st.pyplot(fig_2)
            st.pyplot(fig_3)
        elif calculation_mode == "Fat and muscle":
            fig_2, fig_3, axs2, axs3 = predictions_plots_mc(base_file = base_file, PATH_Data = PATH_Data, prediction_image_path = path_pred)
            st.pyplot(fig_2)
            st.pyplot(fig_3)             
if app_mode == "Animation":
    st.header('Animation')
    st.subheader('Fat and muscle prediction with MobileNetV2')
    if st.checkbox('Show Animation'):
# Animation for the training, for now it is only the same image, but should change
        st.subheader('Training evolution')
        base_file = "00202.png"
        image_1 = st.empty()
        slider_ph = st.empty()
        info_ph = st.empty()
        path_pred = Get_prediction_path(path_model = "", base_file = base_file, epoch = 0,
                    Animation = True)   
        fig_2, fig_3, axs2, axs3 = predictions_plots_mc(base_file = base_file, PATH_Data = PATH_Data, prediction_image_path = path_pred)
        image_1.pyplot(fig_2)
#        image_2.pyplot(fig_3)
        epoch = slider_ph.slider("Training evolution", 0, 120, 0, 10)
        if st.button('Animate'):
################## We could include a condition here if we only have in one model implemented###################            
            for x in range(0, 120, 10):
                epoch = slider_ph.slider("Training evolution", 0, 120, x+10, 10)        
                path_pred = Get_prediction_path(path_model = path_model, base_file = base_file, epoch = epoch,
                    Animation = True)        
                fig_2, fig_3, axs2, axs3 = predictions_plots_mc(base_file = base_file, PATH_Data = PATH_Data, prediction_image_path = path_pred)
                image_1.pyplot(fig_2)
#                image_2.pyplot(fig_3) 
    
elif app_mode == "Prediction mode": # Here would be posible to use different models
    st.header('Prediction Mode')
    st.subheader(calculation_mode + " prediction with " + model_type + " model")
    uploaded_file = st.empty()
    if calculation_mode == "Fat":
        uploaded_file.empty()
        uploaded_file = st.file_uploader("Choose an image", type =['png', 'jpg'])
        try:
            model = load_model(path_model, SAVE_FORMAT, custom_metrics)
            fig_5, axs5 = predict_plot_binary(uploaded_file, W, H, PLOTTING_WIDTH,PLOTTING_HEIGHT, num_classes, model)
            image_5 = st.pyplot(fig_5)
        except:
#            st.sidebar.success('To continue load an image')
            st.success('To continue load an image')
    elif calculation_mode == "Fat and muscle":
        uploaded_file.empty()
        uploaded_file = st.file_uploader("Choose an image", type =['png', 'jpg'])
        try:
            model = load_model(path_model, SAVE_FORMAT, custom_metrics)    
            fig_5, axs5 = predict_plot_multiclass(uploaded_file, W, H, PLOTTING_WIDTH,PLOTTING_HEIGHT, num_classes, model) 
            image_5 = st.pyplot(fig_5) 
        except:
#            st.sidebar.success('To continue load an image')
            st.success('To continue load an image')
    
