import streamlit as st
import onnxruntime as rt
import numpy as np
from PIL import Image

# --- 1. LOAD THE ONNX MODEL ---
@st.cache_resource
def load_model():
    session = rt.InferenceSession('dfu_model.onnx')
    return session

session = load_model()
class_names = ['Abnormal(Ulcer)', 'Normal(Healthy skin)']

# --- 2. PURE NUMPY PREPROCESSING (Replaces TensorFlow's preprocess_input) ---
def preprocess_image(image):
    # EfficientNet expects pixels scaled to [-1, 1] instead of [0, 255]
    img_array = np.array(image, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    return img_array

# --- 3. APP UI ---
st.set_page_config(page_title="DFU Classifier", page_icon="🩺", layout="centered")
st.title("🩺 Diabetic Foot Ulcer (DFU) Classifier")
st.write("Upload a 224x224 patch image to check if it is **Normal** or **Abnormal**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.write("### Analyzing Image...")
        
        # --- 4. PREPROCESS ---
        img = image.resize((224, 224))
        img = img.convert('RGB') # Ensure 3 channels even if image is PNG with transparency
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_image(img_array) # Use our math function instead of TF
        
        # --- 5. PREDICT USING ONNX ---
        with st.spinner('Running CNN...'):
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img_array})[0]
            
        # --- 6. INTERPRET RESULT ---
        # ONNX outputs a 2D array, e.g., [[0.12, 0.88]] for 2 classes
        prediction_prob = outputs[0][1] # Probability of Class 1 (Normal)
        
        if prediction_prob < 0.5:
            predicted_class = class_names[0]
            confidence = (1 - prediction_prob) * 100
            st.error(f"**Prediction:** {predicted_class}")
        else:
            predicted_class = class_names[1]
            confidence = prediction_prob * 100
            st.success(f"**Prediction:** {predicted_class}")
            
        st.info(f"**Model Confidence:** {confidence:.2f}%")
