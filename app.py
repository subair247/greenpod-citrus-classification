import os
import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.classical_model import extract_citrus_features
from src.dl_model import build_mobilenet_classifier

st.set_page_config(page_title="Greenpod Labs - Citrus Quality Classifier", page_icon="🍊", layout="centered")

st.title("🍊 Greenpod Labs - Produce Quality Vision")
st.markdown("### Citrus Quality & Post-Harvest Defect Classifier")
st.write("Compare **Approach 1 (Classical CIELAB+LBP+SVM)** vs **Approach 2 (MobileNetV2 Transfer Learning)** in real time.")

CLASS_NAMES = ["blackspot", "canker", "fresh", "grenning"]
CLASS_DESCRIPTIONS = {
    "fresh": "Healthy, export-grade fresh orange with uniform skin pigmentation.",
    "grenning": "Citrus Greening (Huanglongbing) chlorosis symptoms with green-yellow mottling.",
    "blackspot": "Citrus Black Spot (Phyllosticta citricarpa) necrotic circular lesions.",
    "canker": "Citrus Canker (Xanthomonas citri) raised, erumpent corky pustules."
}

# Model loading with caching
@st.cache_resource
def load_models():
    svm_path = "models/svm_classifier.joblib"
    scaler_path = "models/feature_scaler.joblib"
    dl_path = "models/mobilenet_v2.pth"
    
    svm_model, scaler, dl_model = None, None, None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(svm_path) and os.path.exists(scaler_path):
        svm_model = joblib.load(svm_path)
        scaler = joblib.load(scaler_path)
        
    if os.path.exists(dl_path):
        dl_model = build_mobilenet_classifier(num_classes=4).to(device)
        dl_model.load_state_dict(torch.load(dl_path, map_location=device))
        dl_model.eval()
        
    return svm_model, scaler, dl_model, device

svm_model, scaler, dl_model, device = load_models()

uploaded_file = st.file_uploader("Upload Citrus Fruit Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temporary file for OpenCV extraction
    temp_path = "temp_uploaded_img.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded Citrus Image", use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # 1. Classical SVM Inference
    with col1:
        st.subheader("Approach 1: Classical SVM")
        if svm_model and scaler:
            feat = extract_citrus_features(temp_path).reshape(1, -1)
            feat_scaled = scaler.transform(feat)
            probs = svm_model.predict_proba(feat_scaled)[0]
            pred_idx = np.argmax(probs)
            label = CLASS_NAMES[pred_idx]
            conf = probs[pred_idx] * 100
            
            st.metric(label="Predicted State", value=label.upper())
            st.write(f"**Confidence:** `{conf:.2f}%`")
            st.info(CLASS_DESCRIPTIONS[label])
        else:
            st.warning("SVM weights not found in `models/`")

    # 2. Deep Learning MobileNetV2 Inference
    with col2:
        st.subheader("Approach 2: MobileNetV2")
        if dl_model:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = transform(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = dl_model(tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                pred_idx = np.argmax(probs)
                label = CLASS_NAMES[pred_idx]
                conf = probs[pred_idx] * 100
                
            st.metric(label="Predicted State", value=label.upper())
            st.write(f"**Confidence:** `{conf:.2f}%`")
            st.info(CLASS_DESCRIPTIONS[label])
        else:
            st.warning("MobileNetV2 weights not found in `models/`")

    if os.path.exists(temp_path):
        os.remove(temp_path)
