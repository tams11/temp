import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# Load the pre-trained model (no need to train it again)
model = load_model('model_skin.h5')  # Ensure 'model_skin.h5' is in your working directory or adjust path

# Streamlit app interface
st.title("Skin Classification")
st.header("Please upload a skin image or take a picture with your camera")

# List of class labels for prediction
class_labels = ["AD", "Normal", "Others"]

# File uploader in Streamlit
camera_file = st.camera_input("", label_visibility="collapsed", key="camera_input")
uploaded_file = st.file_uploader("or choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

image_placeholder = st.empty()
image_source = None

# Process the image based on user input
if uploaded_file is not None:
    image_source = Image.open(uploaded_file).convert("RGB")
    image_placeholder.image(image_source, caption="Uploaded Image", use_column_width=True, key="uploaded_image")
elif camera_file is not None:
    image_source = Image.open(camera_file).convert("RGB")
    image_placeholder.image(image_source, caption="Captured Image", use_column_width=True, key="captured_image")

# If an image is uploaded or captured, classify it
if image_source is not None:
    st.write("Classifying...")

    # Image preprocessing
    def preprocess(image):
        image = image.resize((224, 224))  # Adjust the size as needed
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    
    # Preprocess the image before feeding it to the model
    processed_image = preprocess(image_source)

    # Predict the class of the uploaded image
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_label = class_labels[predicted_class]

    # Display the prediction result
    st.write(f"Prediction: {class_label}")

# Optional: Clear image after prediction
if uploaded_file or camera_file:
    st.session_state["uploaded_image"] = None
    st.session_state["captured_image"] = None
