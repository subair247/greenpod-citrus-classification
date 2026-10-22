import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import joblib
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.classical_model import extract_citrus_features
from src.dl_model import build_mobilenet_classifier

CLASS_NAMES = ["blackspot", "canker", "fresh", "grenning"]

def run_inference(image_path: str, svm_path: str = "models/svm_classifier.joblib",
                  scaler_path: str = "models/feature_scaler.joblib",
                  dl_path: str = "models/mobilenet_v2.pth"):
    if not os.path.exists(image_path):
        print(f"Error: Target image file not found at '{image_path}'")
        return

    print("=" * 65)
    print(f"GREENPOD LABS - CITRUS QUALITY & DEFECT DETECTION")
    print(f"Input Target Image: {image_path}")
    print("=" * 65)

    # 1. Approach 1: Classical Machine Learning (SVM)
    if os.path.exists(svm_path) and os.path.exists(scaler_path):
        svm_model = joblib.load(svm_path)
        scaler = joblib.load(scaler_path)
        features = extract_citrus_features(image_path).reshape(1, -1)
        features_scaled = scaler.transform(features)
        svm_probs = svm_model.predict_proba(features_scaled)[0]
        svm_idx = np.argmax(svm_probs)
        svm_label = CLASS_NAMES[svm_idx]
        svm_conf = svm_probs[svm_idx] * 100
        print(f"[*] Approach 1 (Classical SVM):  {svm_label.upper():<10} [Confidence: {svm_conf:.2f}%]")
    else:
        print("[!] Approach 1 Model weights not found in models/ folder.")

    # 2. Approach 2: Deep Learning (MobileNetV2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if os.path.exists(dl_path):
        model = build_mobilenet_classifier(num_classes=4).to(device)
        model.load_state_dict(torch.load(dl_path, map_location=device))
        model.eval()

        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            dl_idx = np.argmax(probs)
            dl_label = CLASS_NAMES[dl_idx]
            dl_conf = probs[dl_idx] * 100
            print(f"[*] Approach 2 (MobileNetV2):   {dl_label.upper():<10} [Confidence: {dl_conf:.2f}%]")
    else:
        print("[!] Approach 2 Model weights not found in models/ folder.")

    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Run Citrus Quality & Defect Inference Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input citrus image file")
    args = parser.parse_args()
    run_inference(args.image)
