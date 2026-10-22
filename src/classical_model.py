import os
import glob
from typing import Tuple, List
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def extract_citrus_features(image_path: str, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Extracts concatenated CIELAB color histograms and Local Binary Pattern (LBP)
    texture descriptors from an input citrus image.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read image at: {image_path}")
        
    img_resized = cv2.resize(img_bgr, target_size)
    
    # 1. CIELAB Color Space (L* lightness, a* green-red axis, b* blue-yellow axis)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256])
    hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256])
    hist_b = cv2.calcHist([lab], [2], None, [16], [0, 256])
    color_features = np.concatenate([hist_l, hist_a, hist_b]).flatten()
    
    # 2. Local Binary Pattern (LBP) Texture Descriptor
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    
    # Normalization
    color_features = color_features / (np.sum(color_features) + 1e-7)
    lbp_hist = lbp_hist.astype(np.float32) / (np.sum(lbp_hist) + 1e-7)
    
    return np.concatenate([color_features, lbp_hist])

def train_svm_pipeline(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[SVC, StandardScaler]:
    """
    Standardizes features and fits a Support Vector Machine (RBF kernel).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)
    return clf, scaler
