#!/usr/bin/env python
# coding: utf-8
#dosyayı py olarak kaydet ve komut satırını kullanarak streamlit run streamlit.py 
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
model=load_model('rice_class_cnn.h5')
def process_image(img):
    img=img.resize((224,224))
    img=np.array(img)
    img=img/255.0
    img=np.expand_dims(img,axis=0)
    return img
st.title('Rice Classification')
st.write('Upload a rice pic and let it predict!')
file=st.file_uploader('Bir resim seç', type= ['jpg','jpeg','png'])
class_names=['Arborio','Basmati','Ipsala','Jasmine','Karacadag']    
if file is not None:
    img=Image.open(file)
    st.image(img,caption='yuklenen resim')
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)
    st.write(prediction)
    st.write('I think its : ',class_names[predicted_class])