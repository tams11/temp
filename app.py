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
    
def main():
    
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    with st.sidebar:
        st.markdown(
        """
        <div class="css-1d391kg">
        <h3>Navigation</h3>
        """,
        unsafe_allow_html=True
    )
        if st.button("Home ►", key="home_button"):
            st.session_state.page = "home"
        if st.button("Camera (Skin Classification) ►", key="camera_button"):
            st.session_state.page = "camera"
        if st.button("Manual SCORAD Input ►", key="manual_input_button"):
            st.session_state.page = "manual_input"

    # Render selected page
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "camera":
        camera_page()
    elif st.session_state.page == "manual_input":
        manual_input_page()

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
        <p>Tinanggal ko yung ano ah</p>
        <p>"Get Started" na button</p>
        <p>Diko kasi alam pano icenter nadadamay yung buttons sa gilid</p>
    </div>
    """, unsafe_allow_html=True)
        
def camera_page():
    st.title("Skin Classification")
    st.markdown("""
    <div class="content-section">
        <h2>📸 Upload a skin image or take a picture with your camera</h2>
    </div>
    """, unsafe_allow_html=True)
    class_labels = ["Mild", "Moderate", "Severe"]
    severity_scores = {"Mild": 3, "Moderate": 6, "Severe": 9}

    # Initialize B1 in session state if it doesn't exist
    if 'B1' not in st.session_state:
        st.session_state.B1 = 0

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
        severity_score = severity_scores[class_label]
        
        # Store the severity score in B1
        st.session_state.B1 = severity_score
        
        st.write(f"Prediction: {class_label}")
        st.write(f"Severity Score (B1): {st.session_state.B1}")
        
        # Display a colored box based on severity
        if class_label == "Mild":
            st.markdown(f'<div style="background-color: #90EE90; padding: 10px; border-radius: 5px;">Mild Case (B1 = {st.session_state.B1})</div>', unsafe_allow_html=True)
        elif class_label == "Moderate":
            st.markdown(f'<div style="background-color: #FFD700; padding: 10px; border-radius: 5px;">Moderate Case (B1 = {st.session_state.B1})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #FF6B6B; padding: 10px; border-radius: 5px;">Severe Case (B1 = {st.session_state.B1})</div>', unsafe_allow_html=True)

    if uploaded_file or camera_file:
        st.session_state["uploaded_image"] = None
        st.session_state["captured_image"] = None
        
def manual_input_page():
    st.title("Manual SCORAD Input")
    st.header("Enter the SCORAD Intensity Information")

    # Initialize B1 in session state if it doesn't exist
    if 'B1' not in st.session_state:
        st.session_state.B1 = 0

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("### Severity Guide")
        st.markdown("""
        - Mild: B1 = 3
        - Moderate: B1 = 6
        - Severe: B1 = 9
        """)

    with col2:
        st.write("### Input")
        severity = st.selectbox("Select Severity Level:", ["Mild", "Moderate", "Severe"])
        st.session_state.B1 = {"Mild": 3, "Moderate": 6, "Severe": 9}[severity]
        
        if st.button("Submit"):
            st.success(f"Severity Level: {severity}")
            st.success(f"B1 Score: {st.session_state.B1}")

    with col3:
        st.write("### Result")
        if 'severity' in locals():
            if severity == "Mild":
                st.markdown(f'<div style="background-color: #90EE90; padding: 10px; border-radius: 5px;">Mild Case (B1 = {st.session_state.B1})</div>', unsafe_allow_html=True)
            elif severity == "Moderate":
                st.markdown(f'<div style="background-color: #FFD700; padding: 10px; border-radius: 5px;">Moderate Case (B1 = {st.session_state.B1})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background-color: #FF6B6B; padding: 10px; border-radius: 5px;">Severe Case (B1 = {st.session_state.B1})</div>', unsafe_allow_html=True)

# Run the main app

        
if __name__ == "__main__":
    main()