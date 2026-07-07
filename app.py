import streamlit as st
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# --- 1. LOAD THE TFLITE MODEL ---
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path='dfu_model.tflite')
    interpreter.allocate_tensors() # Required to initialize the model
    return interpreter

interpreter = load_model()

# Get input and output details from the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Abnormal(Ulcer)', 'Normal(Healthy skin)']

# --- 2. PURE NUMPY PREPROCESSING ---
def preprocess_image(image):
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
        img = img.convert('RGB')
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_image(img_array)
        
        # --- 5. PREDICT USING TFLITE ---
        with st.spinner('Running CNN...'):
            # Set the image as the input tensor
            interpreter.set_tensor(input_details[0]['index'], img_array)
            # Run the inference
            interpreter.invoke()
            # Get the output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
        # --- 6. INTERPRET RESULT ---
        prob = output_data[0][0] 
        
        if prob < 0.5:
            predicted_class = class_names[0]
            confidence = (1 - prob) * 100
            st.error(f"**Prediction:** {predicted_class}")
        else:
            predicted_class = class_names[1]
            confidence = prob * 100
            st.success(f"**Prediction:** {predicted_class}")
            
        st.info(f"**Model Confidence:** {confidence:.2f}%")
