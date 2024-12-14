import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the custom CSS file
load_css("style.css")

# Add custom object scope to handle DepthwiseConv2D compatibility
@tf.keras.utils.register_keras_serializable()
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        if 'groups' in config:
            del config['groups']
        return config

# Register the custom layer
tf.keras.utils.get_custom_objects().update({
    'DepthwiseConv2D': CustomDepthwiseConv2D
})

# Load the model with custom objects
model = tf.keras.models.load_model('keras_model.h5', compile=False)

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
        <p>In compliance with the Data Privacy Act of 2012 (Republic Act No. 10173), all data collected in this study will be used solely for the purposes outlined in the research objectives. The researchers guarantee that no collected data will be reused for any other study or dataset beyond the scope of this project.  

Upon completion of the research, all data will be securely disposed of in accordance with Section 11(e) of the Act, which states that personal information must be retained only for as long as necessary for the fulfillment of the purposes for which it was obtained. This ensures the data subjects' rights to privacy and confidentiality are upheld.  

All collected data will be permanently deleted and rendered unrecoverable to prevent unauthorized access, in line with the proper disposal practices mandated by the law.</p>
    </div>
    """, unsafe_allow_html=True)
        
def camera_page():
    st.title("Skin Classification")
    st.markdown("""
    <div class="content-section">
        <h2>📸 Upload a skin image or take a picture with your camera</h2>
    </div>
    """, unsafe_allow_html=True)
    class_labels = ["Mild", "Moderate", "Severe", "Normal"]
    severity_scores = {"Mild": 3, "Moderate": 6, "Severe": 9, "Normal": 0}

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
            # Teachable Machine preprocessing
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            return data

        processed_image = preprocess(image_source)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])
        class_label = class_labels[predicted_class]
        confidence_score = prediction[0][predicted_class]
        
        # Score ng B1
        st.session_state.B1 = severity_scores[class_label]
        st.session_state.image_processed = True
        
        if class_label == "Normal":
            st.markdown(f'<div style="background-color: #808080; padding: 10px; border-radius: 5px;">Intensity Level: {class_label} (Confidence: {confidence_score:.2%})</div>', unsafe_allow_html=True)
        elif class_label == "Mild":
            st.markdown(f'<div style="background-color: #90EE90; padding: 10px; border-radius: 5px;">Intensity Level: {class_label} (Confidence: {confidence_score:.2%})</div>', unsafe_allow_html=True)
        elif class_label == "Moderate":
            st.markdown(f'<div style="background-color: #FFD700; padding: 10px; border-radius: 5px;">Intensity Level: {class_label} (Confidence: {confidence_score:.2%})</div>', unsafe_allow_html=True)
        elif class_label == "Severe":
            st.markdown(f'<div style="background-color: #FF6B6B; padding: 10px; border-radius: 5px;">Intensity Level: {class_label} (Confidence: {confidence_score:.2%})</div>', unsafe_allow_html=True)
            
        if class_label != "Normal":
            st.write("Please proceed to Manual SCORAD Input to complete the assessment.")
            if st.button("Proceed to Manual SCORAD Input", type="primary"):
                st.session_state.page = "manual_input"

    if uploaded_file or camera_file:
        st.session_state["uploaded_image"] = None
        st.session_state["captured_image"] = None
        
def manual_input_page():
    # Initialize session state variables
    if 'A' not in st.session_state:
        st.session_state.A = 0
    if 'B1' not in st.session_state:
        st.session_state.B1 = 0
    if 'B2_swelling' not in st.session_state:
        st.session_state.B2_swelling = 0
    if 'B2_thickening' not in st.session_state:
        st.session_state.B2_thickening = 0
    if 'B2_dryness' not in st.session_state:
        st.session_state.B2_dryness = 0
    if 'C' not in st.session_state:
        st.session_state.C = 0

    main_content, rubrics = st.columns([0.6, 0.4])

    with main_content:
        st.title("Manual SCORAD Input")
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
            if not st.session_state.get('image_processed', False):
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

            # Part B2 - Individual Intensity Criteria
            st.subheader("Part B2 - Additional Intensity Criteria")
            
            # B2 inputs in columns with consistent styling
            st.markdown("""
                <div class="b2-input-container">
                    <div class="b2-input-column">
                        <h5>Swelling (0-3)</h5>
            """, unsafe_allow_html=True)
            
            B2_swelling = st.number_input(
                "Swelling",
                min_value=0,
                max_value=3,
                value=int(st.session_state.B2_swelling),
                help="Rate the severity of swelling from 0 (none) to 3 (severe)",
                label_visibility="collapsed",
                key="swelling"
            )
            
            st.markdown("""
                    </div>
                    <div class="b2-input-column">
                        <h5>Thickening (0-3)</h5>
            """, unsafe_allow_html=True)
            
            B2_thickening = st.number_input(
                "Thickening",
                min_value=0,
                max_value=3,
                value=int(st.session_state.B2_thickening),
                help="Rate the severity of skin thickening from 0 (none) to 3 (severe)",
                label_visibility="collapsed",
                key="thickening"
            )
            
            st.markdown("""
                    </div>
                    <div class="b2-input-column">
                        <h5>Dryness (0-3)</h5>
            """, unsafe_allow_html=True)
            
            B2_dryness = st.number_input(
                "Dryness",
                min_value=0,
                max_value=3,
                value=int(st.session_state.B2_dryness),
                help="Rate the severity of skin dryness from 0 (none) to 3 (severe)",
                label_visibility="collapsed",
                key="dryness"
            )
            
            st.markdown("""
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Calculate total B2 score
            B2_total = B2_swelling + B2_thickening + B2_dryness

            # Part C - Subjective symptoms
            st.subheader("Part C - Subjective Symptoms")
            C_input = st.number_input(
                "Input score for Part C (Itchiness and Sleep Loss) of SCORAD test (0-20):",
                min_value=0,
                max_value=20,
                value=int(st.session_state.C)
            )

            # Remove SCORAD interpretation from here
            submitted = st.form_submit_button("Calculate Total SCORAD", use_container_width=True)

            if submitted:
                # Update session state values
                st.session_state.A = A_input
                if not st.session_state.get('image_processed', False):
                    st.session_state.B1 = B1_input
                st.session_state.B2_swelling = B2_swelling
                st.session_state.B2_thickening = B2_thickening
                st.session_state.B2_dryness = B2_dryness
                st.session_state.C = C_input

                # Convert to float for calculation
                A = float(st.session_state.A)
                B1 = float(st.session_state.B1)
                B2 = float(B2_total)
                C = float(st.session_state.C)
                
                # SCORAD calculation formula: A/5 + 7(B1+B2)/2 + C
                total_scorad = (A / 5) + (7 * (B1 + B2) / 2) + C
                
                # Display individual B2 scores
                st.write("### B2 Scores Breakdown:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Swelling", B2_swelling)
                with col2:
                    st.metric("Thickening", B2_thickening)
                with col3:
                    st.metric("Dryness", B2_dryness)
                st.write(f"Total B2 Score: {B2_total}")
                
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
                
                # Add clinic recommendation
                st.markdown("""
                <div class="content-section">
                    <h2>Recommended Clinic for Consultation</h2>
                    <p><strong>Marichu's Derma Clinic</strong><br>
                    Address: Ground Floor, Unit 7, Jacinto Building, Molino Rod 3,<br>
                    Bacoor Cavite, Philippines, 4102</p>
                </div>
                """, unsafe_allow_html=True)

    # Rubrics column
    with rubrics:
        st.markdown("### Scoring Guidelines")
        
        with st.expander("Area of Extent (Part A)", expanded=True):
            st.markdown("""
            | Percentage | Description | Guidelines |
            |------------|-------------|------------|
            | 0% | No area affected | No visible signs of atopic dermatitis. |
            | 1-10% | Minimal area affected | Small patches or localized areas (e.g., a small portion of the face, one elbow, or one hand). |
            | 11-30% | Mildly affected area | Multiple small areas affected or a slightly larger region (e.g., a leg or part of the torso). |
            | 31-50% | Moderately affected area | Significant portions of the body affected, covering a large part of the arms, legs, or torso. |
            | 51-70% | Severely affected area | Over half the body surface affected, including multiple large areas (e.g., both arms and legs). |
            | 71-100% | Very severely affected, generalized dermatitis | Almost the entire body affected, including widespread involvement of the face, limbs, and trunk. |
            """)

        with st.expander("Intensity Criteria (B1 & B2)", expanded=True):
            st.markdown("""
            
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
            5. Thickening
            6. Dryness
            """)

        with st.expander("Subjective Symptoms (Part C)", expanded=True):
            st.markdown("""
            Patients should evaluate both itchiness and sleep disturbance together, as these factors often influence each other. The score reflects the overall impact of atopic dermatitis on comfort and rest.

            | Score | Description | Guidelines |
            |-------|-------------|------------|
            | 0 | No impact | No itching or sleep disturbance. Sleep is uninterrupted, and there is no discomfort. |
            | 1-5 | Mild impact | Occasional itching with minimal discomfort. Sleep is rarely disturbed and easy to resume after interruptions. |
            | 6-10 | Moderate impact | Frequent itching causing noticeable discomfort. Sleep is occasionally disrupted, with some difficulty returning to sleep. |
            | 11-15 | Severe impact | Persistent and intense itching that significantly disrupts daily comfort. Sleep is frequently disturbed, leading to fatigue or reduced daytime productivity. |
            | 16-20 | Very severe impact | Constant, unbearable itching causing extreme discomfort. Sleep is severely impaired or nearly impossible, leading to exhaustion and a serious impact on life. |
            """)

        # SCORAD Index Interpretation moved to the bottom
        with st.expander("SCORAD Index Interpretation", expanded=True):
            st.markdown("""
            | SCORAD Score | Severity Level | Description |
            |--------------|----------------|-------------|
            | < 25 | Mild | • Limited areas affected (typically 1-10% body surface) • Mild redness and dryness • Minimal swelling and thickening • Occasional itching with minimal sleep disturbance • Limited impact on daily activities |
            | 25-50 | Moderate | • Multiple affected areas (typically 11-50% body surface) • Noticeable redness and dryness • Moderate swelling and skin thickening • Frequent itching causing some sleep disruption • Moderate impact on quality of life |
            | > 50 | Severe | • Widespread involvement (>50% body surface) • Intense redness, significant dryness • Severe swelling and skin thickening • Persistent itching with major sleep disturbance • Significant impact on daily activities and quality of life |
            """)

if __name__ == "__main__":
    main()