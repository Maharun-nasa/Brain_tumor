# =============================================
# app.py - Streamlit App for Brain Tumor Detection
# =============================================

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="🧠 Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# --- Load Model and Class Names (cached for performance) ---
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model('best_model.h5')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, CLASS_NAMES = load_model_and_classes()
IMG_SIZE = (224, 224)

# --- UI Header ---
st.title("🧠 Brain Tumor Detection AI")
st.markdown("""
Upload an MRI scan to detect:
- Glioma (Malignant brain tumor)
- Meningioma (Mostly benign tumor)
- Pituitary (Hormone gland tumor)
- No Tumor (Healthy brain)
""")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an MRI image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI scan in JPG, JPEG, or PNG format."
)

# --- Prediction Logic ---
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    
    # Preprocess image
    with st.spinner("Analyzing the MRI scan..."):
        image = image.resize(IMG_SIZE)
        img_array = np.array(image) / 255.0
        img_array = img_array.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        pred_probs = model.predict(img_array)[0]
        pred_class_idx = np.argmax(pred_probs)
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = float(pred_probs[pred_class_idx])
    
    # --- Display Results ---
    st.divider()
    st.subheader("📊 Diagnosis Result")
    
    if pred_class == "notumor":
        st.success(f"Prediction: {pred_class.upper()} ✅")
        st.info("The scan appears healthy. No tumor detected.")
    else:
        st.error(f"Prediction: {pred_class.upper()} ⚠️")
        st.warning(f"A {pred_class} tumor has been detected. Please consult a medical professional.")
    
    st.metric("Confidence", f"{confidence:.2%}")
    
    # --- Show all probabilities as progress bars ---
    st.subheader("📊 Detailed Probabilities")
    for cls, prob in zip(CLASS_NAMES, pred_probs):
        st.progress(float(prob), text=f"{cls}: {prob:.2%}")