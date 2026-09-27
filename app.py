import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgriGuard | Potato Disease Intelligence",
    page_icon="🥔",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- MODERN UI / CSS STYLING ---
st.markdown("""
    <style>
    /* Main background & font styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        padding: 2.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .hero-container h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hero-container p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }

    /* Result metric badges */
    .result-badge {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.25rem;
        margin-top: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.875rem;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTENT ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/potato.png", width=80)
    st.markdown("### **AgriGuard AI**")
    st.write("An advanced deep learning diagnostic pipeline built to safeguard crop yields by detecting early signs of foliage pathology.")
    
    st.markdown("---")
    st.markdown("#### **Diagnostic Classes**")
    st.markdown("🟢 **Healthy:** Normal leaf structure.")
    st.markdown("🟡 **Early Blight:** Target-like necrotic spots.")
    st.markdown("🔴 **Late Blight:** Water-soaked dark lesions.")
    
    st.markdown("---")
    st.info("Tip: For best results, upload a clear, well-lit photo of a single potato leaf against a plain background.")

# --- LOAD MODEL WITH CACHING ---
@st.cache_resource
def load_prediction_model():
    # Loads the potatoes.h5 model securely from root directory
    model = tf.keras.models.load_model("potatoes.h5")
    return model

with st.spinner("Initializing neural network... Please wait..."):
    model = load_prediction_model()

# --- MAIN APP INTERFACE ---
st.markdown("""
    <div class="hero-container">
        <h1>🥔 Potato Disease Intelligence</h1>
        <p>Instant computer vision diagnostics for agricultural health monitoring</p>
    </div>
""", unsafe_allow_html=True)

# File uploader container
uploaded_file = st.file_uploader("Upload Leaf Specimen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("### **Uploaded Specimen**")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### **Diagnostic Panel**")
        st.write("Ready to analyze cellular patterns and surface conditions.")
        
        analyze_btn = st.button("Run Diagnostics", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Extracting features and evaluating patterns..."):
            # Preprocess image to model specifications (256x256)
            image_resized = image.resize((256, 256))
            img_array = np.array(image_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)
            class_names = ["Early Blight", "Late Blight", "Healthy"]
            
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = np.max(predictions[0]) * 100
            
            st.markdown("---")
            st.markdown("### **Analysis Complete**")
            
            # Dynamic color coding based on health status
            if predicted_class == "Healthy":
                st.success(f"Diagnosis: **{predicted_class}**\n\nConfidence: **{confidence:.2f}%**")
            elif predicted_class == "Early Blight":
                st.warning(f"Diagnosis: **{predicted_class}**\n\nConfidence: **{confidence:.2f}%**")
            else:
                st.error(f"Diagnosis: **{predicted_class}**\n\nConfidence: **{confidence:.2f}%**")
                
            # Confidence Breakdown Bar Chart
            st.write("Class Probability Distribution:")
            chart_data = {
                class_names[i]: float(predictions[0][i]) for i in range(len(class_names))
            }
            st.bar_chart(chart_data)

else:
    st.info("👆 Upload an image using the button above to begin diagnostics.")

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        Powered by TensorFlow, Streamlit & Computer Vision • Developed for Smart Agriculture
    </div>
""", unsafe_allow_html=True)
