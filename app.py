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
import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = "home"

def sidebar_navigation():
    st.markdown('<h1 class="css-1d391kg">Navigation</h1>', unsafe_allow_html=True)

    # Custom navigation buttons
    if st.button("Home", key="home_button"):
        st.session_state.page = "home"
    if st.button("Camera (Skin Classification)", key="camera_button"):
        st.session_state.page = "camera"
    if st.button("Manual SCORAD Input", key="manual_input_button"):
        st.session_state.page = "manual_input"


def home_page():
    st.title("Welcome to Atopic Dermatitis Intensity Interpretation")
    st.markdown("""
    <div class="content-section">
    <h2>Learn More About Atopic Dermatitis</h2>
    <p>TESTING!  \nKuha ako description dun sa paper mismo?</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-section">
        <h2>Data Privacy Act</h2>
        <p>AYWAN KO!  \nDIKO ALAM TO!  \n  \t:)  \t</p>
    </div>
    """, unsafe_allow_html=True)
       
    if st.button("Get Started"):
        st.session_state.page = "camera"
        
def camera_page():
    st.title("Skin Classification")
    st.markdown("""
    <div class="content-section">
        <h2>📸 Upload a skin image or take a picture with your camera</h2>
    </div>
    """, unsafe_allow_html=True)
    class_labels = ["AD", "Normal", "Others"]

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
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_label = class_labels[predicted_class]
        st.write(f"Prediction: {class_label}")
    if uploaded_file or camera_file:
        st.session_state["uploaded_image"] = None
        st.session_state["captured_image"] = None
        
def manual_input_page():
    st.title("Manual SCORAD Input")
    st.header("Enter the SCORAD Intensity Information")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("### Guide")
        st.markdown("DUMIDIKIT SA GITNA")

    with col2:
        st.write("### Input")
        scorad_value = st.text_input("Enter SCORAD Value: ")
        if st.button("Submit"):
            if scorad_value.isdigit():
                st.success(f"SCORAD Value Submitted: {scorad_value}")
            else:
                st.error("Invalid input. Please enter a valid number.")

    with col3:
        st.write("### Result")
        st.write("Submitted SCORAD results will be displayed here. kung gagana HAHAHAH")

# Run the main app
def main():
    st.sidebar.markdown('<div class="css-1d391kg">Navigation</div>', unsafe_allow_html=True)
    sidebar_navigation()

    # Render selected page
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "camera":
        camera_page()
    elif st.session_state.page == "manual_input":
        manual_input_page()
        
if __name__ == "__main__":
    main()