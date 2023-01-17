# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:51:26 2023

@author: Sahil
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

st.title('Potatoes disease detection')

#upload image from local system
image = st.file_uploader('Test Image', type = ['png', 'jpg'])

#convert image into array & display it
img_array = mpimg.imread(image)
st.image(img_array)

#load deep learning model
loaded_model = load_model(r'C:\Users\Sahil\DS ML tests\Project Potato Disease Classification\potatoes_disease_detection.h5')

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

#predict the labels & the confidence %
img_array = tf.expand_dims(img_array, 0)
pred = loaded_model.predict(img_array)[0]
pred_index = np.argmax(pred)
pred_label = class_names[pred_index]
confidence = round(100*np.max(pred), 2)

st.write(f"Predicted: {pred_label}")
st.write(f"Confidence: {confidence}%")
