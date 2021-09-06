# Importing Libraries
import streamlit as st
import io
import numpy as np
from PIL import Image 
import tensorflow as tf
import json
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import efficientnet.tfkeras as efn

# Title and Description
st.title('Bangladesh Local Spinach Recognition')
st.write("Just Upload Your Spinach Image and Get Predictions")
st.write("")


gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


model = tf.keras.models.load_model("D:/projects/spinach/Webapp/Xception-spinach-model.h5")


uploaded_file = st.file_uploader("Choose a Image file", type=["png", "jpg"])


predictions_map = {0:"Jute Spinach", 1:"Malabar Spinach", 2:"Red Spinach", 3:"Taro Spinach", 4:"Water Spinach"}

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, use_column_width=True)
    resized_image = np.array(image.resize((224, 224)))/255.
    image_batch = resized_image[np.newaxis, :, :, :]
    predictions_arr = model.predict(image_batch)
    predictions = np.argmax(predictions_arr)
    result_text = f"The Spinach Is {predictions_map[predictions]} With {int(predictions_arr[0][predictions]*100)}% Probability"
    if predictions == 0:
        st.success(result_text)
    else:
        st.error(result_text)
           

