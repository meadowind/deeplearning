#Live Demo to run this "streamlit run app.py"

import streamlit as st
import tensorflow as tf
from src.preprocess import process_image
import numpy as np

st.title("Traffic Sign Recognition Demo")
model = tf.keras.models.load_model('models/baseline.h5')

uploaded_file = st.file_uploader("Upload a traffic sign image...", type=["jpg", "png", "ppm"])

if uploaded_file is not None:
    # Save temp file to process
    with open("temp.ppm", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    img = process_image("temp.ppm")
    st.image(img, caption='Processed Image', use_column_width=True)
    
    # Predict
    pred = model.predict(np.expand_dims(img, axis=0))
    class_id = np.argmax(pred)
    st.success(f"Predicted Class: {class_id} (Confidence: {np.max(pred)*100:.2f}%)")