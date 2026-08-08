import os
import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from preprocess import preprocess_image_pytorch, extract_hsv_color_analysis
from model import build_mango_cnn_model, CLASS_NAMES, CLASS_DISPLAY_NAMES
from rule_engine import evaluate_mango_decision_rules

st.set_page_config(
    page_title="AI Mango Quality & Ripeness System",
    page_icon="🥭",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #D97706;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        text-align: center;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .rule-box {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = 'mango_model.pth'

@st.cache_resource
def load_trained_model():
    """Loads the pre-trained PyTorch CNN model."""
    if os.path.exists(MODEL_PATH):
        try:
            model = build_mango_cnn_model()
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading PyTorch model: {e}")
            return None
    return None

def main():
    st.markdown("<h1 class='main-title'>🥭 AI-Based Mango Quality & Ripeness Assessment System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'><b>CS22032 Essentials of Artificial Intelligence</b> | Neural Network + Computer Vision + Rule Engine Solution</p>", unsafe_allow_html=True)
    
    model = load_trained_model()
    
    st.sidebar.header("⚙️ Control Panel & Settings")
    base_price = st.sidebar.number_input(
        "Base Market Price per kg (Grade A Benchmark):",
        min_value=50.0,
        max_value=5000.0,
        value=300.0,
        step=25.0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Applied AI Concepts")
    st.sidebar.markdown("""
    1. **Convolutional Neural Network (CNN)**: Deep learning pattern recognition for quality grading (*Fresh/Grade A*, *Unripe/Grade B*, *Spoiled/Grade C*).
    2. **Computer Vision (OpenCV)**: HSV color space extraction for skin color & decay spot analysis.
    3. **Rule-Based Expert System**: Inference engine for shelf-life estimation & price recommendations.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📸 Choose Input Method")
    input_choice = st.sidebar.radio(
        "Select Image Source:",
        ["Upload Your Own Mango Image", "Use Preset Sample Demo Image"]
    )
    
    selected_image = None
    
    if input_choice == "Upload Your Own Mango Image":
        uploaded_file = st.sidebar.file_uploader("Upload a Mango image (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
    else:
        sample_dir = 'test_images'
        sample_options = {
            "Sample 1: Grade A Ripe Mango 🥭": os.path.join(sample_dir, 'sample_ripe_mango.jpg'),
            "Sample 2: Grade B Unripe Green Mango 🍏": os.path.join(sample_dir, 'sample_unripe_mango.jpg'),
            "Sample 3: Grade C Overripe / Damaged Mango 🍂": os.path.join(sample_dir, 'sample_overripe_mango.jpg')
        }
        selected_sample_label = st.sidebar.selectbox("Select Sample Image:", list(sample_options.keys()))
        sample_path = sample_options[selected_sample_label]
        
        if os.path.exists(sample_path):
            selected_image = Image.open(sample_path)
        else:
            st.warning("Sample image files not found yet. Please run `python train.py` first!")

    if selected_image is not None:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("📷 Mango Input Image")
            st.image(selected_image, use_container_width=True, caption="Uploaded / Selected Input Image")
            
            # Preprocess image for PyTorch
            tensor_batch, img_rgb = preprocess_image_pytorch(selected_image)
            
            # Computer Vision Feature Extraction
            color_features = extract_hsv_color_analysis(img_rgb)
            
            st.markdown("#### 🔬 CV Feature Extraction (HSV Analysis)")
            cv_col1, cv_col2, cv_col3 = st.columns(3)
            cv_col1.metric("Ripe Yellow %", f"{color_features['yellow_percentage']}%")
            cv_col2.metric("Unripe Green %", f"{color_features['green_percentage']}%")
            cv_col3.metric("Dark Spots %", f"{color_features['dark_spots_percentage']}%")
            
        with col2:
            st.subheader("🧠 AI Neural Network & Decision Engine")
            
            if model is None:
                st.warning("⚠️ Model weights file (`mango_model.pth`) not found. Please run `python train.py` to train!")
                y_pct = color_features['yellow_percentage']
                g_pct = color_features['green_percentage']
                d_pct = color_features['dark_spots_percentage']
                
                if d_pct > 3.0:
                    pred_class = 'Grade_C_Overripe'
                    conf = 0.88
                elif g_pct > y_pct:
                    pred_class = 'Grade_B_Unripe'
                    conf = 0.91
                else:
                    pred_class = 'Grade_A_Ripe'
                    conf = 0.95
            else:
                with torch.no_grad():
                    logits = model(tensor_batch)
                    probabilities = F.softmax(logits, dim=1).numpy()[0]
                    pred_index = int(np.argmax(probabilities))
                    pred_class = CLASS_NAMES[pred_index]
                    conf = float(probabilities[pred_index])
                
            rule_res = evaluate_mango_decision_rules(pred_class, conf, base_price_per_kg=base_price)
            
            st.markdown(f"### **Result: {CLASS_DISPLAY_NAMES[pred_class]}**")
            st.progress(min(1.0, conf))
            st.caption(f"Neural Network Classification Confidence: **{rule_res['confidence_percentage']}%**")
            
            st.markdown("---")
            st.markdown("### 📋 Expert Decision Support & Pricing Advisory")
            
            m1, m2 = st.columns(2)
            m1.metric("Recommended Price / kg", f"Rs. {rule_res['recommended_price']}", f"-{rule_res['discount_percentage']}% Discount" if rule_res['discount_percentage'] > 0 else "Full Price")
            m2.metric("Est. Remaining Shelf Life", rule_res['estimated_shelf_life'])
            
            st.markdown(f"""
            <div class="rule-box">
                <h4><b>💡 Vendor Storage & Sales Strategy</b></h4>
                <p><b>Status:</b> {rule_res['status_category']}</p>
                <p><b>Recommendation:</b> {rule_res['vendor_recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.info("👈 Please select or upload a Mango image from the sidebar to begin AI quality evaluation.")

if __name__ == "__main__":
    main()
