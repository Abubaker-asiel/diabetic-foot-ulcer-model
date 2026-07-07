import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="DFU Classifier", page_icon="🩺", layout="centered")

# --- 2. LOAD THE MODEL ---
# @st.cache_resource ensures the model only loads into memory once, making the app fast
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('dfu_cnn_model.keras')
    return model

model = load_model()

# These MUST match the exact names from your Kaggle training output
# (image_dataset_from_directory sorts alphabetically)
class_names = ['Abnormal(Ulcer)', 'Normal(Healthy skin)']

# --- 3. APP UI ---
st.title("🩺 Diabetic Foot Ulcer (DFU) Classifier")
st.write("Upload a 224x224 patch image to check if it is **Normal** or **Abnormal**.")

# File uploader (accepts jpg, png, jpeg)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.write("### Analyzing Image...")
        
        # --- 4. PREPROCESS THE IMAGE ---
        # 1. Resize to 224x224 (if it isn't already)
        img = image.resize((224, 224))
        # 2. Convert to numpy array
        img_array = np.array(img)
        # 3. Add a batch dimension (model expects shape: [1, 224, 224, 3])
        img_array = np.expand_dims(img_array, axis=0)
        # 4. Apply the EXACT same preprocessing used during training
        img_array = preprocess_input(img_array)
        
        # --- 5. MAKE PREDICTION ---
        with st.spinner('Running CNN...'):
            # Predict returns a probability between 0 and 1 because of 'sigmoid'
            prediction_prob = model.predict(img_array)[0][0]
            
        # --- 6. INTERPRET RESULT ---
        # 0.0 to 0.49 = Class 0 (Abnormal), 0.50 to 1.0 = Class 1 (Normal)
        if prediction_prob < 0.5:
            predicted_class = class_names[0]
            confidence = (1 - prediction_prob) * 100
            st.error(f"**Prediction:** {predicted_class}")
        else:
            predicted_class = class_names[1]
            confidence = prediction_prob * 100
            st.success(f"**Prediction:** {predicted_class}")
            
        st.info(f"**Model Confidence:** {confidence:.2f}%")