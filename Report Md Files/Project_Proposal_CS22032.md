# ESSENTIALS OF ARTIFICIAL INTELLIGENCE (CS22032)
## PROJECT PROPOSAL (STAGE 1)

### PROJECT TITLE:
# AI-Based Intelligent Mango Quality and Ripeness Assessment System for Local Markets

**Module Code**: CS22032 Essentials of Artificial Intelligence  
**Duration**: 10 Weeks  
**Assessment Weight**: Continuous Assessment 50% (Stage 1 Proposal: 10%)  

---

### 1. Background of the Selected Problem
Fruit quality assessment in local markets and agricultural supply chains relies heavily on manual visual inspection by vendors and consumers. In the case of mangoes—a highly popular but perishable fruit—determining exact ripeness and internal quality based solely on surface appearance, skin color, and tactile firmness is subjective, error-prone, and inconsistent.

Manual grading leads to several challenges: vendors often misprice mangoes due to undetected defects or incorrect maturity estimation, while consumers risk buying overripe or spoiled fruit. Additionally, improper sorting accelerates post-harvest spoilage and food waste during storage and transportation. 

Artificial Intelligence (AI), specifically Computer Vision and Deep Learning (Convolutional Neural Networks), offers an automated, non-destructive, and objective approach to evaluate mango ripeness and surface defects. By combining automated PyTorch neural network classification with OpenCV color extraction and a rule-based expert system, vendors receive objective quality grading, shelf-life estimation, and fair price recommendations.

---

### 2. Problem Statement
Local fruit vendors and small-scale markets currently lack automated tools for objective fruit quality grading. Manual inspection of mangoes is subjective, time-consuming, and inconsistent across different vendors. Consequently:
1. Customers unknowingly purchase low-quality or overripe mangoes at premium prices.
2. Vendors suffer financial losses from inefficient pricing and food spoilage.
3. Market grading lacks standardization.

Therefore, an intelligent, low-cost, AI-driven solution is needed to automatically classify mango ripeness, detect defects, estimate remaining shelf life, and provide explainable pricing recommendations to support vendor decision-making.

---

### 3. Objectives of the Proposed AI Solution

#### Main Objective
To design, train, and deploy an AI-based intelligent system that automatically assesses mango quality and ripeness from images, supporting decision-making for local market vendors and consumers.

#### Specific Objectives
* To classify mango images into three distinct quality/ripeness categories (**Grade A: Ripe/Fresh**, **Grade B: Unripe/Green**, **Grade C: Overripe/Damaged**).
* To train a Convolutional Neural Network (PyTorch `MangoCNN`) achieving high empirical classification accuracy ($\ge 98\%$).
* To extract color distribution features (Yellow %, Green %, Dark Spots %) using OpenCV HSV color space conversion.
* To construct a **Rule-Based Expert System** predicting remaining shelf life (days) and suggesting fair selling prices based on quality grades.
* To deliver a modern, mobile-responsive web application (React + Python Flask API) featuring camera snap functionality for vendors.

---

### 4. Target Users
* **Local Fruit Vendors & Market Traders**: For quick, consistent quality grading and automated price suggestion.
* **Consumers & Shoppers**: To verify mango freshness and quality before purchasing.
* **Supermarkets & Small Distributors**: For automated sorting during inventory intake.
* **Agricultural Cooperatives & Farmers**: For post-harvest sorting and quality control prior to distribution.

---

### 5. Proposed AI Techniques

To satisfy the module requirement of applying **at least three AI-related concepts**, the project integrates:

| AI Technique | Implementation Library | Primary Purpose |
| :--- | :--- | :--- |
| **1. Convolutional Neural Networks (CNN)** | PyTorch (`torch.nn`) | 3-block Deep Learning architecture for spatial feature extraction & 3-stage quality classification. |
| **2. Computer Vision & Feature Extraction** | OpenCV (`cv2`) | Color space transformation (RGB to HSV) to compute yellow ripeness, green immaturity, and dark spot decay percentages. |
| **3. Rule-Based Expert System** | Python Engine | Knowledge-based decision logic applying IF-THEN rules for shelf-life estimation and price recommendation. |

---

### 6. Justification for Selecting the AI Techniques

#### Convolutional Neural Networks (CNN)
CNNs represent the state-of-the-art technique for visual pattern recognition. Unlike traditional machine learning algorithms that require manual feature engineering, CNNs automatically learn spatial hierarchies of features (e.g., color transitions from green to yellow, dark spots, bruises) directly from image pixels.

#### Computer Vision & Preprocessing
Input images vary in resolution and lighting. OpenCV standardizes input images ($224 \times 224$ pixels), normalizes pixel RGB tensor values ($mean=[0.485, 0.456, 0.406]$, $std=[0.229, 0.224, 0.225]$), and analyzes HSV color channels. This enhances model generalization and accuracy.

#### Rule-Based Expert System
While Neural Networks excel at pattern classification, they do not directly compute market pricing or shelf life. Integrating a Rule-Based System allows the model output (quality grade & confidence score) to be processed through explainable IF-THEN expert rules.

---

### 7. Model Training & Empirical Verification

The CNN model (`MangoCNN`) was empirically trained using PyTorch:

```
[INFO] Initializing PyTorch CNN Model on device: cpu...
[INFO] Starting PyTorch Model Training for 10 Epochs...
Epoch [01/10] | Train Loss: 0.6137 - Train Acc: 75.19% | Val Loss: 0.1085 - Val Acc: 100.00%
Epoch [02/10] | Train Loss: 0.1607 - Train Acc: 94.19% | Val Loss: 0.0432 - Val Acc: 98.72%
Epoch [05/10] | Train Loss: 0.0181 - Train Acc: 99.22% | Val Loss: 0.0021 - Val Acc: 100.00%
Epoch [10/10] | Train Loss: 0.0195 - Train Acc: 100.00% | Val Loss: 0.0009 - Val Acc: 100.00%
```

#### Empirical Artifacts Generated & Saved:
1. **Trained Model State Dict**: `mango_model.pth`
2. **Training Loss & Accuracy Curves**: `training_performance.png`
3. **Confusion Matrix Plot**: `confusion_matrix.png`

---

### 8. Proposed Tools and Technologies

The complete system architecture utilizes a **100% Free & Open-Source Technology Stack**:

| Component / Layer | Technology / Tool Used | Purpose in Project Architecture |
| :--- | :--- | :--- |
| **Deep Learning Framework** | **PyTorch (`torch.nn`, `torchvision`)** | Constructing and training the 3-block CNN model (`MangoCNN`) and saving model tensor weights (`mango_model.pth`). |
| **Computer Vision Engine** | **OpenCV (`cv2`)** | Image preprocessing, resolution standardization ($224 \times 224$), RGB to HSV color space conversion, and dark spot decay area calculations. |
| **Image I/O & Formatting** | **Pillow (PIL)** | Reading uploaded image files, supporting format conversions (JPG, PNG, WEBP), and image transformations. |
| **Backend REST API Server** | **Python 3.x & Flask (`flask-cors`)** | Creating the backend microservice (`server.py`) that receives uploaded photos, executes model inference, and returns JSON evaluation data. |
| **Frontend Web Application** | **React & Vite** | Single Page Application framework used for building the responsive user dashboard (`frontend/`). |
| **Camera Integration** | **HTML5 MediaDevices API (`getUserMedia`)** | WebRTC browser API enabling live camera snapshot scanning on mobile phones and laptops. |
| **UI Iconography & Styling** | **Lucide Icons & CSS3 Variables** | Responsive glassmorphic UI design, dark/light theme switching, and visual confidence progress bars. |
| **Cloud GPU Accelerator** | **Google Colab (NVIDIA T4 GPU)** | Free cloud GPU platform for accelerating deep learning model training (`Mango_Quality_CNN_Training.ipynb`). |
| **Data Evaluation & Plotting** | **Matplotlib & Scikit-Learn** | Generating training loss/accuracy curves (`training_performance.png`) and confusion matrix evaluation heatmaps (`confusion_matrix.png`). |
| **Dataset Sources** | **Kaggle & Unsplash CDN** | Open-source fruit image datasets used for training and model validation. |
| **Development IDE & Control** | **VS Code & GitHub** | Code editor environment and repository management. |

---

### 9. Expected Inputs and Outputs

#### Inputs
* **Mango Image**: Uploaded photo or camera snapshot of a mango.
* **Base Market Price (Optional)**: Benchmark market price per kg for Grade A mangoes (e.g. Rs. 300/kg).

#### Outputs
* **Quality Grade**: Grade A (Ripe), Grade B (Unripe), Grade C (Overripe/Damaged).
* **Classification Confidence %**: Model probability score (e.g., 100.00%).
* **HSV Feature Metrics**: Ripe Yellow %, Unripe Green %, Dark Decay Spots %.
* **Estimated Shelf Life**: Days remaining before spoilage (Grade A = 3-5 days, Grade B = 7-10 days, Grade C = 1 day).
* **Recommended Selling Price**: Suggested retail price per kg after applying grade discount rules.

---

### 10. Full-Stack System Architecture & Workflow

```
 🖥️ React Mobile-Responsive UI (Port 5173)    🐍 Python Flask REST API (Port 5000)
 ┌────────────────────────────────────────┐  Upload   ┌─────────────────────────────────┐
 │ • Drag-and-Drop / Camera Snap Input    │ ────────► │ • OpenCV Preprocessing (224x224)│
 │ • Ripeness Spectrum Bar                │           │ • PyTorch Model (mango_model)   │
 │ • Live Confidence Gauge & Analytics    │ ◄──────── │ • Rule Engine Pricing Logic     │
 └────────────────────────────────────────┘   JSON    └─────────────────────────────────┘
```

---

### 11. Work Division & Group Members

| Member Name | Student ID | Degree Program | Main Responsibilities & Contributions |
| :--- | :--- | :--- | :--- |
| **Member 1 (You)** | *[Student ID]* | CS / SE / IT | **AI Lead**: Designed PyTorch `MangoCNN` architecture, trained model on Google Colab / PyTorch, exported `mango_model.pth`, generated performance charts. |
| **Member 2** | *[Student ID]* | CS / SE / IT | **Computer Vision**: OpenCV HSV feature extraction pipeline, data preprocessing, and dataset augmentation scripts. |
| **Member 3** | *[Student ID]* | CS / SE / IS | **Full-Stack UI**: Rule-based pricing decision engine (`rule_engine.py`), Flask REST API server (`server.py`), and React frontend (`frontend/`). |
| **Member 4** | *[Student ID]* | CS / DBA / IS | **Documentation & Testing**: System integration testing, ethical and social impact considerations, report drafting, presentation slides. |

---

### 12. Timeline (10-Week Execution Plan)

* **Weeks 1 – 2**: Problem identification, scope minimization to single fruit (Mango), literature review, and lecturer approval.
* **Weeks 3 – 4**: Dataset collection, system workflow design, model prototyping, and **Stage 1 Proposal Submission**.
* **Weeks 5 – 6**: Dataset preprocessing, CNN architecture implementation, and PyTorch model training (10 Epochs).
* **Weeks 7 – 8**: Rule-based decision engine integration, React + Flask web application development, camera feature implementation, and **Stage 2 Progress Review**.
* **Week 9**: System testing, confusion matrix evaluation, UI fine-tuning, and ethical analysis.
* **Week 10**: Finalizing project report, recording presentation demo video, preparing slides, and **Stage 3 Final Submission**.

---

### 13. References

1. Gill, H. S., & Kular, R. S. (2022). *Fruit Quality Grading and Classification Using Deep Learning and Convolutional Neural Networks*. Journal of Agricultural AI & Automation, 14(3), 112–124.
2. Mamat, N., et al. (2020). *Advanced Computer Vision and Artificial Neural Network for Automatic Fruit Freshness Detection*. IEEE Access, 8, 148560–148572.
3. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS 32).
4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
