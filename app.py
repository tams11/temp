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
def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Sidebar for Navigation
    with st.sidebar:
        st.title("Navigation")
        if st.button("Home"):
            st.session_state.page = "home"
        if st.button("Camera (Skin Classification)"):
            st.session_state.page = "camera"
        if st.button("Manual SCORAD Input"):
            st.session_state.page = "manual_input"

    # Render pages based on the state
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "camera":
        camera_page()
    elif st.session_state.page == "manual_input":
        manual_input_page()


def home_page():
    # Display a title and some information
    st.markdown(
        """
        <div class="home-page">
            <h1>Welcome to Atopic Dermatitis Monitoring</h1>
            <div class="section learn-more">
                <h3>Learn More About Atopic Dermatitis</h3>
                <p>Atopic Dermatitis (AD) is a common chronic skin condition...</p>
            </div>
            <div class="section data-privacy">
                <h3>Data Privacy Act</h3>
                <p>This system complies with the Data Privacy Act...</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Custom "Get Started" button
    st.markdown(
        """
        <style>
        .custom-button {
            display: block;
            margin: 30px auto;
            padding: 15px 30px;
            background-color: #007BFF;
            color: white;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            text-decoration: none;
            border: none;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            cursor: pointer;
        }
        .custom-button:hover {
            background-color: #0056b3;
        }
        </style>
        <a href="#" onclick="window.location.href = '/?page=camera';" class="custom-button">Get Started</a>
        """,
        unsafe_allow_html=True,
    )



def camera_page():
    st.title("Skin Classification")
    st.header("📸 Upload a skin image or take a picture with your camera")
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
        st.markdown("Follow the instructions to manually input SCORAD values.")

    with col2:
        st.write("### Input")
        scorad_value = st.text_input("Enter SCORAD Value:")
        if st.button("Submit"):
            if scorad_value.isdigit():
                st.success(f"SCORAD Value Submitted: {scorad_value}")
            else:
                st.error("Invalid input. Please enter a valid number.")

    with col3:
        st.write("### Result")
        st.write("Submitted SCORAD results will be displayed here.")

st.markdown(
    """
    <script>
    function navigateTo(page) {
        // Send the page name to Streamlit using session state
        var myInput = window.parent.document.querySelectorAll('[data-testid="stMarkdownContainer"] input')[0];
        myInput.value = page;
        myInput.dispatchEvent(new Event('input', { bubbles: true }));
    }
    </script>
    """,
    unsafe_allow_html=True,
)

# Add a hidden input to capture page navigation
st.text_input("hidden_page_state", key="page", label_visibility="hidden")

# Run the main app
if __name__ == "__main__":
    main()