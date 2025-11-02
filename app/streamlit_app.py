import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

st.set_page_config(page_title="Waste Classifier", page_icon="♻️")
st.title("♻️ Waste Sorting Classifier (Simple)")

MODEL_PATH = "models/best_model.h5"
model = load_model(MODEL_PATH)

class_names = ["E-waste","automobile wastes","battery waste", "glass waste", "light bulbs" , "metal waste", "organic waste" , "paper waste", "plastic waste"]

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224,224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Predict
    print(model.output_shape)

    preds = model(img_tensor)
    class_id = tf.argmax(preds[0]).numpy()
    conf = tf.reduce_max(preds[0]).numpy()

    st.subheader(f"Prediction: **{class_names[class_id]}** ({conf*100:.2f}%)")
