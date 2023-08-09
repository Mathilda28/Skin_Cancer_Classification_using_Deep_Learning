import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title='Predict',
    layout='wide',
    initial_sidebar_state='expanded'
)

#load model
best_model = load_model('model2.h5')


def img_predict(img, model):
    pred = np.array(img)[:, :, :3]
    pred = tf.image.resize(pred, size=(240, 240))
    pred = pred / 255.0

    
    predicted_probabilities = model.predict(x=tf.expand_dims(pred, axis=0))[0]

    
    predicted_class_index = np.argmax(predicted_probabilities)

    
    if predicted_class_index == 0:
        return "benign"
    else:
        return "malignant"




    

def run():
    # variable image
    img = None

    # Image upload and prediction
    uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        prediction = img_predict(img, best_model)

        # Display the prediction result
        title = f"<h2 style='text-align:center'>{prediction}</h2>"
        st.markdown(title, unsafe_allow_html=True)
        st.image(img, use_column_width=True)

if __name__ == "__main__":
    run()