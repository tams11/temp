import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import streamlit as st

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the custom CSS file
load_css("style.css")

class HubLayer(tf.keras.layers.Layer):
    def __init__(self, handle, trainable=False, **kwargs):
        super(HubLayer, self).__init__(trainable=trainable, **kwargs)
        self.handle = handle

    def build(self, input_shape):
        self.hub_layer = hub.KerasLayer(self.handle, trainable=self.trainable)
        # This assumes the Hub layer has a single output.
        # You might need to adjust this if it has multiple outputs.
        self.hub_layer.build(input_shape)
        # Instead of directly appending to _trainable_weights and _non_trainable_weights,
        # add the weights using self.add_weight, but replace '/' with '_' in names
        for w in self.hub_layer.trainable_weights:
            self.add_weight(
                name=w.name.replace('/', '_'),  # Replace '/' with '_'
                shape=w.shape,
                dtype=w.dtype,
                trainable=True
            )
        for w in self.hub_layer.non_trainable_weights:
            self.add_weight(
                name=w.name.replace('/', '_'),  # Replace '/' with '_'
                shape=w.shape,
                dtype=w.dtype,
                trainable=False
            )
        super(HubLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.hub_layer(inputs, **kwargs)
    
    def get_config(self):
        config = super(HubLayer, self).get_config()
        config.update({"handle": self.handle})
        return config

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, k, name=None, **kwargs):
        super(CustomLayer, self).__init__(name=name)
        self.k = k
        super(CustomLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({"k": self.k})
        return config

    def call(self, input):
        return tf.multiply(input, 2)

tf.keras.utils.get_custom_objects().update({'HubLayer': HubLayer})

model = tf.keras.models.load_model('model_skin.h5', custom_objects={'HubLayer': HubLayer, 'CustomLayer': CustomLayer})

# Streamlit app interface
st.title("Skin Classification")
st.header(" 📸 Upload a skin image or take a picture with your camera ")

class_labels = ["AD", "Normal", "Others"]

# File uploader in Streamlit
camera_file = st.camera_input("", label_visibility="collapsed", key="camera_input")
uploaded_file = st.file_uploader("or choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

image_placeholder = st.empty()
image_source = None

if uploaded_file is not None:
    image_source = Image.open(uploaded_file).convert("RGB")
    image_placeholder.image(image_source, caption="Uploaded Image", use_column_width=True)
elif camera_file is not None:
    image_source = Image.open(camera_file).convert("RGB")
    image_placeholder.image(image_source, caption="Captured Image", use_column_width=True)

if image_source is not None:
    st.write("Classifying...")
 
    def preprocess(image):

        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    
    processed_image = preprocess(image_source)
    
    # Predict the class of the uploaded image
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_label = class_labels[predicted_class]

    # Display the prediction result
    st.write(f"Prediction: {class_label}")

if uploaded_file or camera_file:
    st.session_state["uploaded_image"] = None
    st.session_state["captured_image"] = None