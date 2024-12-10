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
        self.hub_layer.build(input_shape)
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
        </div>
        """,
        unsafe_allow_html=True
    )
        col1, = st.columns(1)
        with col1:
            if st.button("Home", key="home_button", use_container_width=True):
                st.session_state.page = "home"
            if st.button("Camera (AD Intensity)", key="camera_button", use_container_width=True):
                st.session_state.page = "camera"
            if st.button("Manual SCORAD Input", key="manual_input_button", use_container_width=True):
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
        <p>Atopic Dermatitis, also called atopic eczema, is the most common inflammatory skin disease worldwide. AD is a significant public health concern due to the immune system's instability in AD, there is chronic inflammation and a compromised skin barrier, which causes severe pain, itching, and an increased risk of infection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-section">
        <h2>Data Privacy Act</h2>
        <p>Tinanggal ko yung ano ah</p>
        <p>"Get Started" na button</p>
        <p></p>
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

    # Initialize B1 variable
    if 'B1' not in st.session_state:
        st.session_state.B1 = 0
    if 'image_processed' not in st.session_state:
        st.session_state.image_processed = False

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
        
        # Score ng B1
        st.session_state.B1 = severity_score
        st.session_state.image_processed = True
        
        if class_label == "Mild":
            st.markdown(f'<div style="background-color: #90EE90; padding: 10px; border-radius: 5px;">Intensity Level: {class_label}</div>', unsafe_allow_html=True)
        elif class_label == "Moderate":
            st.markdown(f'<div style="background-color: #FFD700; padding: 10px; border-radius: 5px;">Intensity Level: {class_label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #FF6B6B; padding: 10px; border-radius: 5px;">Intensity Level: {class_label}</div>', unsafe_allow_html=True)
            
        st.write("Please proceed to Manual SCORAD Input to complete the assessment.")

    if uploaded_file or camera_file:
        st.session_state["uploaded_image"] = None
        st.session_state["captured_image"] = None
        
def manual_input_page():
    main_content, rubrics = st.columns([0.6, 0.4])

    # Main content column
    with main_content:
        st.title("Manual SCORAD Input")
        st.header("Complete SCORAD Assessment")

        # Initialize variables in session state if they don't exist
        if 'A' not in st.session_state:
            st.session_state.A = 0
        if 'B1' not in st.session_state:
            st.session_state.B1 = 0
        if 'B2' not in st.session_state:
            st.session_state.B2 = 0
        if 'C' not in st.session_state:
            st.session_state.C = 0
        if 'image_processed' not in st.session_state:
            st.session_state.image_processed = False
        
        # Create a form for all inputs
        with st.form("scorad_form"):
            # Part A - Area
            st.subheader("Part A - Area of Extent Score")
            A_input = st.number_input(
                "Input score for Part A (Extent) of SCORAD test (0-100):",
                min_value=0,
                max_value=100,
                value=int(st.session_state.A)
            )

            # Part B1 - Only show if no image was processed
            if not st.session_state.image_processed:
                st.subheader("Part B1 - Intensity Score")
                B1_input = st.number_input(
                    "Input score for Part B1 of SCORAD test (0-9):",
                    min_value=0,
                    max_value=9,
                    value=int(st.session_state.B1)
                )
            else:
                st.subheader("Part B1 - Intensity Score")
                st.info(f"B1 score from image analysis: {st.session_state.B1}")
                B1_input = st.session_state.B1

            # Part B2 - Intensity
            st.subheader("Part B2 - Additional Intensity Criteria")
            B2_input = st.number_input(
                "Input score for Part B2 of SCORAD test (0-9):",
                min_value=0,
                max_value=9,
                value=int(st.session_state.B2)
            )

            # Part C - Subjective symptoms
            st.subheader("Part C - Subjective Symptoms")
            C_input = st.number_input(
                "Input score for Part C (Pruritus and Sleep Loss) of SCORAD test (0-20):",
                min_value=0,
                max_value=20,
                value=int(st.session_state.C)
            )

            # Submit button
            submitted = st.form_submit_button("Calculate Total SCORAD")

            if submitted:
                # Update session state values
                st.session_state.A = A_input
                if not st.session_state.image_processed:
                    st.session_state.B1 = B1_input
                st.session_state.B2 = B2_input
                st.session_state.C = C_input

                # Convert to float for calculation to maintain precision
                A = float(st.session_state.A)
                B1 = float(st.session_state.B1)
                B2 = float(st.session_state.B2)
                C = float(st.session_state.C)
                
                # SCORAD calculation formula: A/5 + 7(B1+B2)/2 + C
                total_scorad = (A / 5) + (7 * (B1 + B2) / 2) + C
                st.success(f"Total SCORAD Score: {total_scorad:.2f}")
                
                # Interpret SCORAD score
                if total_scorad < 25:
                    severity = "Mild"
                    color = "#90EE90"
                elif total_scorad < 50:
                    severity = "Moderate"
                    color = "#FFD700"
                else:
                    severity = "Severe"
                    color = "#FF6B6B"
                    
                st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px;">SCORAD Assessment: {severity} ({total_scorad:.2f})</div>', unsafe_allow_html=True)

    # Rubrics column
    with rubrics:
        st.markdown("### Scoring Guidelines")
        
        with st.expander("Area of Extent (Part A)", expanded=True):
            st.markdown("""
            | % Area | Description |
            |--------|-------------|
            | 0% | No area affected |
            | 1-10% | Minimal area |
            | 11-30% | Mildly affected |
            | 31-50% | Moderately affected |
            | 51-70% | Severely affected |
            | 71-100% | Very severely affected |
            """)

        with st.expander("Intensity Criteria (B1 & B2)", expanded=True):
            st.markdown("""
            Score each from 0-3:
            - 0: None
            - 1: Mild
            - 2: Moderate
            - 3: Severe

            **B1 Criteria:**
            1. Redness
            2. Oozing/crusting
            3. Scratchmarks

            **B2 Criteria:**

            4. Swelling
            5. Skin
            6. Dryness
            """)

        with st.expander("Subjective Symptoms (Part C)", expanded=True):
            st.markdown("""
            | Score | Impact |
            |-------|---------|
            | 0 | No impact |
            | 1-5 | Mild |
            | 6-10 | Moderate |
            | 11-15 | Severe |
            | 16-20 | Very severe |

            *Combine itchiness and sleep disturbance (0-20)*
            """)

if __name__ == "__main__":
    main()