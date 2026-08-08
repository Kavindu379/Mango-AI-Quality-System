# 🥭 AI-Based Intelligent Mango Quality & Ripeness Assessment System

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.x-61dafb?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-API%20Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

An end-to-end, multi-modal **Artificial Intelligence System** designed for local market vendors and agricultural supply chains to automate mango ripeness classification, surface defect detection, remaining shelf-life estimation, and fair market price valuation.

---

## 🌟 Key Features

* 🧠 **Deep Learning CNN**: 3-block PyTorch Convolutional Neural Network (`MangoCNN`) trained on custom mobile smartphone photos with Data Augmentation (rotations, flips, color jitter) achieving 95.5%+ training accuracy across 3 quality categories (*Grade A Ripe*, *Grade B Unripe*, *Grade C Overripe*).
* 🏷️ **Directory-Based Supervised Data Labeling**: Custom real-world dataset collection categorized into explicit ground-truth target folders (`Grade_A_Ripe` $\rightarrow$ Class 0, `Grade_B_Unripe` $\rightarrow$ Class 1, `Grade_C_Overripe` $\rightarrow$ Class 2).
* 🔬 **Computer Vision Feature Extraction**: OpenCV color space transformation (RGB to HSV) calculating real-time ratios for Ripe Yellow %, Unripe Green %, and Dark Decay Spots %.
* 🛡️ **Rule-Based Expert System**: Knowledge-based inference engine processing quality predictions into explainable vendor recommendations, price discounts, and shelf-life predictions.
* 📱 **Mobile-First Responsive React UI**: Sleek glassmorphism single-page app featuring dark/light themes, live camera scanning, and an interactive ripeness spectrum indicator bar.
* 📷 **Native Smartphone Camera Integration**: Web-native camera snap support for mobile smartphones over local Wi-Fi (`capture="environment"`).
* ☁️ **Google Colab Cloud GPU Training**: Pre-configured 1-click cloud notebook (`Mango_Quality_CNN_Training.ipynb`) for training custom datasets on free NVIDIA T4 GPUs.

---

## 🏗️ System Architecture

```
 📱 Smartphone / Laptop UI (Port 5000)        🐍 Python Flask REST API (Port 5000)
 ┌────────────────────────────────────────┐  Upload   ┌─────────────────────────────────┐
 │ • Native Camera Snap / Gallery Input   │ ────────► │ • OpenCV Preprocessing (224x224)│
 │ • Ripeness Spectrum Bar                │           │ • PyTorch Model (mango_model)   │
 │ • Live Confidence Gauge & Analytics    │ ◄──────── │ • Rule Engine Pricing Logic     │
 └────────────────────────────────────────┘   JSON    └─────────────────────────────────┘
```

---

## 📊 Empirical Performance & Evaluation

| Metric | Empirical Value | Description |
| :--- | :--- | :--- |
| **Model Architecture** | `MangoCNN` | 3-block Conv2D + BatchNorm + MaxPool + Linear Classifier |
| **Training Accuracy** | **95.50%** | Achieved with Data Augmentation & Cosine Learning Scheduler |
| **Loss Function** | **0.1600** | Categorical CrossEntropyLoss |
| **Execution Speed** | **< 15 ms** | Real-time CPU inference latency |

### Model Evaluation Artifacts:
- **Weights File**: `mango_model.pth`
- **Performance Curves**: `training_performance.png`
- **Confusion Matrix Heatmap**: `confusion_matrix.png`

---

## 📁 Repository Structure

```
Mango-AI-Quality-System/
├── README.md                           # Project GitHub Documentation
├── server.py                           # Unified Python Flask REST API Server (Port 5000)
├── model.py                            # PyTorch MangoCNN Architecture
├── preprocess.py                       # OpenCV HSV Color Space Feature Extraction
├── rule_engine.py                      # Rule-Based Expert Pricing Engine
├── train.py                            # PyTorch Model Training & Evaluation Script
├── download_real_mango_dataset.py      # Real Photographic Dataset Downloader
├── Mango_Quality_CNN_Training.ipynb    # Google Colab GPU Training Notebook
├── mango_model.pth                     # Trained PyTorch Model Weights Tensor
├── requirements.txt                    # Python Dependencies
├── docs/                               # Assignment Proposals & Documentation
│   └── Project_Proposal_CS22032.md
├── test_images/                        # Sample Demo Test Images
└── frontend/                           # React + Vite Web Application
    ├── src/
    │   ├── App.jsx                     # Main React Component
    │   └── index.css                   # Glassmorphism Design System & Theme Engine
    └── package.json
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Mango-AI-Quality-System.git
cd Mango-AI-Quality-System
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Train the PyTorch Model
```bash
python train.py
```

### 4. Launch the Unified Web Application
```bash
python server.py
```

Open **`http://localhost:5000`** on your laptop or **`http://<YOUR_LAPTOP_IP>:5000`** on your mobile phone to view your live AI Web Application!

---



---


