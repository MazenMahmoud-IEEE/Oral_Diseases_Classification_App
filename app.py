import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
model = tf.keras.models.load_model("best_pretrained_model.keras")

# Class names (same order used during training)
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Preprocessing (same as training)
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype('float32')
    image = preprocess_input(image)  # EfficientNet-style normalization
    return np.expand_dims(image, axis=0)

st.title("🦷 Oral Disease Classification")
st.write("Upload an image of an oral condition and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("Classifying..."):
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        pred_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    st.success(f"🧠 Prediction: **{pred_class}** ({confidence:.2f}%)")
