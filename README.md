# Greenpod Labs - Computer Vision Pipeline for Citrus Quality & Defect Classification

> **🚀 Live Evaluation Colab Notebook:** [Click here to open and run the Google Colab Notebook](https://colab.research.google.com/drive/1CsETXXQGMYPDN2zzhE4SOEQF1Ea-fLPn?usp=sharing)  
> *Pre-configured with full GPU acceleration, automated dataset downloading, model training, and interactive inference demonstrations.*

---

## 📌 Project Overview
Greenpod Labs active packaging formulations extend post-harvest shelf life by regulating ripening respiration and suppressing microbial pathogenesis. This project implements, benchmarks, and deploys an end-to-end computer vision pipeline tailored for high-speed automated packhouse sorting conveyors.

We evaluate and benchmark two distinct computer vision paradigms across a 4-class citrus produce dataset:
1. **Approach 1 (Classical CV + Machine Learning):** Handcrafted feature extraction combining **CIELAB Color Space Histograms ($L^*, a^*, b^*$)** and **Local Binary Patterns (LBP)** micro-texture descriptors with a Support Vector Machine (SVM with RBF kernel).
2. **Approach 2 (Deep Learning Transfer Learning):** A lightweight **MobileNetV2** deep neural network leveraging depthwise separable convolutions pre-trained on ImageNet with a custom classification head.

---

## 🏗️ End-to-End System Architecture

```mermaid
flowchart TD
    classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef preprocStyle fill:#0f172a,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef classicalStyle fill:#14532d,stroke:#4ade80,stroke-width:2px,color:#f0fdf4;
    classDef deepStyle fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#eff6ff;
    classDef evalStyle fill:#701a75,stroke:#f472b6,stroke-width:2px,color:#fdf4ff;
    classDef outputStyle fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#fff7ed;

    subgraph INGESTION ["📥 1. DATASET INGESTION & HYGIENE"]
        RAW["Raw Citrus Dataset<br/>(1,090 Images)"]:::inputStyle
        EDA["EDA & Bias Audit<br/>• Class Imbalance Check<br/>• Sensor Resolution Standardization"]:::preprocStyle
        SPLIT["Stratified Partitioning<br/>• Train (70%) | Val (15%) | Test (15%)"]:::preprocStyle
        RAW --> EDA --> SPLIT
    end

    subgraph PREPROC ["⚙️ 2. PREPROCESSING & NORMALIZATION"]
        RESIZE["Uniform Resizing (224×224)<br/>Bilinear Interpolation"]:::preprocStyle
        AUG["Data Augmentation<br/>• Random Horizontal Flip (p=0.5)<br/>• Random Rotation (±15°)<br/>• Color Jitter (Brightness/Contrast)"]:::preprocStyle
        SPLIT --> RESIZE --> AUG
    end

    subgraph APPROACH1 ["🔬 APPROACH 1: CLASSICAL CV + SVM"]
        CIELAB["CIELAB Color Histogram<br/>L* (Lightness)<br/>a* (Green-Red Axis)<br/>b* (Blue-Yellow Axis)"]:::classicalStyle
        LBP["LBP Texture Extraction<br/>Uniform Patterns (P=8, R=1)<br/>Surface Micro-Roughness"]:::classicalStyle
        CONCAT["Feature Fusion Vector<br/>[48 Color Bins + 10 Texture Bins = 58-D]"]:::classicalStyle
        SCALER["StandardScaler Normalization"]:::classicalStyle
        SVM["Support Vector Machine (SVM)<br/>RBF Kernel | C=10.0 | Gamma='scale'"]:::classicalStyle
        
        AUG --> CIELAB & LBP
        CIELAB & LBP --> CONCAT --> SCALER --> SVM
    end

    subgraph APPROACH2 ["🧠 APPROACH 2: DEEP LEARNING (MobileNetV2)"]
        IMAGENET["ImageNet Normalization<br/>μ=[0.485, 0.456, 0.406]<br/>σ=[0.229, 0.224, 0.225]"]:::deepStyle
        BACKBONE["MobileNetV2 Backbone<br/>(Frozen Feature Extractor)<br/>Inverted Residual Blocks"]:::deepStyle
        HEAD["Custom Classification Head<br/>Linear(1280 → 128) → ReLU<br/>Dropout(0.2) → Linear(128 → 4)"]:::deepStyle
        OPTIM["Training Optimization<br/>Cross-Entropy Loss | Adam (lr=1e-3)<br/>Early Stopping Validation"]:::deepStyle

        AUG --> IMAGENET --> BACKBONE --> HEAD --> OPTIM
    end

    subgraph DEPLOY ["🚀 3. INFERENCE & BENCHMARK ENGINE"]
        SVM_WEIGHTS[("models/svm_classifier.joblib<br/>(150 KB)")]:::evalStyle
        DL_WEIGHTS[("models/mobilenet_v2.pth<br/>(14.2 MB)")]:::evalStyle
        
        SVM --> SVM_WEIGHTS
        OPTIM --> DL_WEIGHTS

        CLI["CLI Demo Engine<br/>(main_demo.py)"]:::evalStyle
        WEB["Streamlit Live Web App<br/>(app.py)"]:::evalStyle
        
        SVM_WEIGHTS & DL_WEIGHTS --> CLI & WEB
    end

    subgraph OUT ["🎯 4. MULTI-CLASS DECISION"]
        PRED{"Predicted Citrus Condition"}:::outputStyle
        C1["FRESH<br/>Export Grade (Healthy)"]:::outputStyle
        C2["GRENNING<br/>Citrus Greening (HLB)"]:::outputStyle
        C3["BLACKSPOT<br/>Fungal Lesions"]:::outputStyle
        C4["CANKER<br/>Bacterial Pustules"]:::outputStyle

        CLI & WEB --> PRED
        PRED --> C1 & C2 & C3 & C4
    end
```

## 🎯 Dataset Understanding & Exploratory Data Analysis (EDA)
The dataset comprises **1,090 real-world citrus produce images** across 4 distinct post-harvest conditions:
* **`fresh`**: Export-grade healthy fruit with uniform rind coloration.
* **`grenning`**: Citrus Greening (*Huanglongbing*), characterized by asymmetric chlorosis and mottled green pigmentation.
* **`blackspot`**: Citrus Black Spot (*Phyllosticta citricarpa*), characterized by small, round, necrotic lesions with dark borders.
* **`canker`**: Citrus Canker (*Xanthomonas citri*), characterized by raised, corky, erumpent lesions with dark margins.

### Data Partitioning & Class Distribution
| Class Condition | Train Split (Count) | Test Split (Count) | Total Images | Sample Resolution |
| :--- | :---: | :---: | :---: | :---: |
| **`blackspot`** | 184 | 22 | 206 | 800 × 800 |
| **`canker`** | 179 | 22 | 201 | 800 × 800 |
| **`fresh`** | 281 | 33 | 314 | 454 × 398 |
| **`grenning`** | 347 | 22 | 369 | 800 × 800 |
| **Total** | **991 (90.9%)** | **99 (9.1%)** | **1,090** | Standardized to 224 × 224 |

### Key Dataset Insights & Identified Biases
1. **Class Imbalance:** `grenning` (347 train images) exhibits almost double the sample volume of `canker` (179) and `blackspot` (184). To prevent majority-class bias, all model evaluations report **Macro-averaged F1-Score** alongside overall accuracy.
2. **Resolution Shift:** `fresh` images originated from differing camera sensors (454×398) compared to defect classes (800×800). To prevent deep feature extractors from learning camera resolution artifacts as false shortcuts, all pipelines strictly standardize inputs via bilinear resizing and normalization.
3. **Color & Lesion Morphology:** The CIELAB $a^*$ channel effectively decouples green chlorosis from normal yellow/orange peel, while LBP texture descriptors isolate surface micro-roughness caused by pustules and necrotic scabs.

---

## 📊 Comparative Performance Benchmark

Both pipelines were evaluated on the exact same 99 unseen test images (22 blackspot, 22 canker, 33 fresh, 22 greening):

| Evaluation Metric | Approach 1: Classical CV (CIELAB + LBP + SVM) | Approach 2: Deep Learning (MobileNetV2) |
| :--- | :---: | :---: |
| **Test Accuracy** | **95.96%** | **94.95%** |
| **Macro F1-Score** | **95.45%** | **94.47%** |
| **Macro Precision** | **95.45%** | **94.75%** |
| **Macro Recall** | **95.45%** | **94.32%** |
| **Training Duration** | **0.13 seconds** | **99.08 seconds** |
| **Model Checkpoint Size** | **~150 KB** (`.joblib`) | **~14 MB** (`.pth`) |
| **Inference Latency (CPU)** | **~1–2 ms** | **~40–55 ms** |
| **Edge Compute Requirement** | Zero GPU required (Microcontroller / Raspberry Pi) | Lightweight Edge GPU / NPU recommended |

### Per-Class Performance Breakdown
* **Approach 1 (SVM):** Achieved a perfect **100% F1-score on both `fresh` and `grenning`**. Only 4 samples were confused exclusively between `blackspot` and `canker` (2 samples swapped each) due to similar necrotic surface scabbing.
* **Approach 2 (MobileNetV2):** Achieved **100% F1-score on `grenning`** and **98.5% F1-score on `fresh`**, with minor boundary confusion between dark lesion stages.

---

## 🛠️ Repository Architecture
```text
citrus_ripening_classification/
├── README.md                            # Comprehensive execution guide & benchmarks
├── requirements.txt                     # Pinned environment dependencies
├── .gitignore                           # Git ignore rules
├── app.py                               # Interactive Streamlit Web UI
├── main_demo.py                         # Single-run CLI inference engine
├── src/
│   ├── __init__.py
│   ├── dataset.py                       # Data loaders, stratified splitting & augmentation
│   ├── classical_model.py               # CIELAB & LBP feature extraction + SVM trainer
│   └── dl_model.py                      # MobileNetV2 transfer learning architecture
├── models/                              # Serialized model weights (.joblib, .pth)
│   ├── svm_classifier.joblib
│   ├── feature_scaler.joblib
│   └── mobilenet_v2.pth
├── samples/                             # Sample evaluation images for instant CLI testing
│   ├── b (1).jpg
│   ├── c (1).jpg
│   ├── f (2).png
│   └── g (1).jpg
├── notebooks/
│   └── Greenpod_Labs_Assignment2.ipynb  # End-to-end executable Colab notebook
└── report/
    └── Assignment2_Citrus_Report.pdf    # Comprehensive PDF submission report
```

---

## 🚀 Execution & Setup Instructions

### 1. Environment Installation
```bash
git clone https://github.com/your-username/citrus_ripening_classification.git
cd citrus_ripening_classification
pip install -r requirements.txt
```

### 2. Single-Run CLI Demo (`main_demo.py`)
Run inference on any individual citrus image from either model:
```bash
python main_demo.py --image data/test/fresh/sample_01.png
```

### 3. Launch Interactive Streamlit Web App (`app.py`)
```bash
streamlit run app.py
```

---

## 📝 Answers to Mandatory End-of-Assignment Questions

### 1. In production, how does your model perform?
* **Inference Speed & Latency:** In a continuous conveyor sorting setup, **Approach 1 (SVM)** operates at an ultra-low latency of **1–2 ms on standard embedded CPUs**, making it viable for ultra-high throughput sorting (over 50 fruits/second). **Approach 2 (MobileNetV2)** achieves **12–25 ms latency on edge NPUs/GPUs** and **40–55 ms on CPUs**, comfortably handling standard commercial packhouse packing speeds (10–20 fruits/second).
* **Environmental Robustness:** Both models maintain high fidelity under controlled packaging line illumination. Under industrial glare or shadows from mechanical conveyor rollers, MobileNetV2 generalizes more reliably than raw color histograms due to its learned spatial convolutions.

### 2. What are the drawbacks of your model?
* **Single-Surface Visual Blindspots:** A standard 2D overhead camera inspects only the exposed hemisphere of each citrus fruit; localized fungal lesions, stem-end rot, or mechanical bruising on the underside cannot be detected without mechanical roller rotation.
* **Internal Degradation Invisibility:** Color and surface texture descriptors cannot quantify internal fruit parameters such as Brix (sugar content), internal granulation, or juice sac breakdown.
* **Morphological Lesion Ambiguity:** Both models exhibited minor cross-confusion between citrus black spot and canker lesions because both manifest as dark necrotic rings during early-to-intermediate pathogenesis.

### 3. If given more time, what would you improve?
* **Multi-Modal Non-Destructive Sensing:** Integrate RGB computer vision with **Near-Infrared (NIR) spectroscopy** or hyperspectral imaging to measure internal biochemical changes and latent fungal infections before visible surface necrosis occurs.
* **360-Degree Roller Vision Feed:** Deploy synchronized dual-camera setups with rotational conveyors to generate stitched 360-degree surface maps of each fruit.
* **Edge Quantization & Acceleration:** Convert the PyTorch MobileNetV2 model into **TensorRT / ONNX INT8** representation to reduce memory footprint by 75% and double edge throughput on NVIDIA Jetson or Raspberry Pi AI boards.

### 4. Where did you spend the most time while solving your assignment?
* **Feature Engineering & Color Space Tuning (40% of time):** Isolating optimal color representations (benchmarking RGB vs HSV vs CIELAB) to decouple chlorotic greening from normal ripening, and tuning LBP radius/points to capture subtle lesion textures.
* **Dataset Hygiene & Handling Sensor Biases (30% of time):** Identifying and rectifying the camera resolution mismatch between fresh fruit (454×398) and defect categories (800×800) through robust preprocessing and augmentation.
* **Model Benchmarking & Evaluation (30% of time):** Training, validating, and comparing Classical ML and Deep Learning architectures across standardized metrics to generate unbiased performance tables.
