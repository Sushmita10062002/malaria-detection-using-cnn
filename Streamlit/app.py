# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 17:49:44 2022

@author: pc
"""

import numpy as np
import os
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
model = load_model('densenet.h5')

def preprocess_image(img):
    img_arr = image.img_to_array(img)
    img_arr = img_arr/255
    img_arr = img_arr.reshape(1,64,64,3)
    return img_arr
    
def prediction(img):
    size = (64,64)
    img = img.resize(size)
    processed_img = preprocess_image(img)
    pred = np.argmax(model.predict(processed_img), axis=1)
    if(pred==0):
        return "Uninfected"
    else:
        return "Infected"

def main():
    st.title("Malaria Detection")
    html_temp = """
    <div style="background-color:#DF7861; padding:5px; margin-bottom:2rem">
    <h3 style="color:white; text-align:center;">Malaria Detection via blood sample images using CNNS</h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    upload_file = st.file_uploader("Upload Cell Images", type="png")
    if st.button("Predict"):
        img = Image.open(upload_file)
        with st.expander('Cell Image', expanded = True):
            st.image(img, use_column_width=True)
        result = prediction(img)
        st.title("Blood sample is {}".format(result))
    
    
if __name__=="__main__":
    main()