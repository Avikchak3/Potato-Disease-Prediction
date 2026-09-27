import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page title
st.set_page_config(page_title="Potato Disease Classification", page_icon="🥔")

@st.cache_resource
def load_model():
    # Load the potatoes.h5 model from your root directory
    model = tf.keras.models.load_model("potatoes.h5")
    return model

with st.spinner("Loading model... Please wait..."):
    model = load_model()

st.title("🥔 Potato Disease Classification")
st.write("Upload an image of a potato leaf to find out if it is healthy, or suffering from Early or Late Blight.")

# File uploader
uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf..."):
            # Resize and preprocess image to match model input (usually 256x256 for this project)
            image = image.resize((256, 256))
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)
            
            # Class names based on the standard codebasics potato dataset structure
            class_names = ["Early Blight", "Late Blight", "Healthy"]
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100
            
            # Display results
            st.markdown(f"### Prediction: **{predicted_class}**")
            st.markdown(f"### Confidence: **{confidence:.2f}%**")