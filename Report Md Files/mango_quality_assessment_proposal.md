# ESSENTIALS OF ARTIFICIAL INTELLIGENCE
**Module Code**: CS22032  
**Document Type**: Project Proposal (Stage 1)  
**Project Title**: AI-Based Intelligent Mango Quality and Ripeness Assessment System for Local Markets  

---

## 1. Background of the Selected Problem
Fruit quality assessment in local markets and agricultural supply chains relies heavily on manual visual inspection by vendors and consumers. In the case of mangoes—a highly popular but perishable fruit—determining exact ripeness and internal quality based solely on surface appearance, skin color, and tactile firmness is subjective, error-prone, and inconsistent.

Manual grading leads to several challenges: vendors often misprice mangoes due to undetected defects or incorrect maturity estimation, while consumers risk buying overripe or spoiled fruit. Additionally, improper sorting accelerates post-harvest spoilage and food waste during storage and transportation. 

Artificial Intelligence (AI), specifically Computer Vision and Deep Learning (Convolutional Neural Networks), offers an automated, non-destructive, and objective approach to evaluate mango ripeness and surface defects. By combining automated neural network classification with a rule-based expert system, vendors can receive objective quality grading, shelf-life estimation, and fair price recommendations.

---

## 2. Problem Statement
Local fruit vendors and small-scale markets currently lack automated tools for objective fruit quality grading. Manual inspection of mangoes is subjective, time-consuming, and inconsistent across different vendors. Consequently:
1. Customers unknowingly purchase low-quality or overripe mangoes at premium prices.
2. Vendors suffer financial losses from inefficient pricing and food spoilage.
3. Market grading lacks standardization.

Therefore, an intelligent, low-cost, AI-driven solution is needed to automatically classify mango ripeness, detect defects, estimate remaining shelf life, and provide explainable pricing recommendations to support vendor decision-making.

---

## 3. Objectives of the Proposed AI Solution

### Main Objective
To design and implement a low-cost, AI-based intelligent system that automatically assesses mango quality and ripeness from images, supporting decision-making for local market vendors and consumers.

### Specific Objectives
* To classify mango images into three distinct quality/ripeness categories (**Grade A: Ripe/Fresh**, **Grade B: Unripe/Green**, **Grade C: Overripe/Damaged**).
* To apply deep learning (CNN) for feature extraction targeting skin color transition and surface defect detection.
* To construct a **Rule-Based Expert System** that predicts remaining shelf life and suggests fair market selling prices based on quality grades.
* To deliver a lightweight, web-based prototype (Streamlit UI) that vendors can easily operate on mobile or desktop devices.

---

## 4. Target Users
* **Local Fruit Vendors & Market Traders**: For quick, consistent quality grading and automated price suggestion.
* **Consumers & Shoppers**: To verify mango freshness and quality before purchasing.
* **Supermarkets & Small Distributors**: For automated sorting during inventory intake.
* **Agricultural Cooperatives & Farmers**: For post-harvest sorting and quality control prior to distribution.

---

## 5. Proposed AI Techniques

To satisfy the module requirement of utilizing **at least three AI-related concepts**, the project integrates:

| AI Technique | Primary Purpose |
| :--- | :--- |
| **1. Convolutional Neural Networks (CNN)** | Deep Learning feature extraction and image classification for mango ripeness grading. |
| **2. Computer Vision & Feature Extraction** | OpenCV image preprocessing, color space conversion (HSV/RGB), and noise reduction. |
| **3. Rule-Based Expert System** | Knowledge-based decision engine for shelf-life estimation and price recommendation rules. |

---

## 6. Justification for Selecting the AI Techniques

### Convolutional Neural Networks (CNN)
CNNs are the state-of-the-art technique for visual pattern recognition. Unlike traditional machine learning algorithms that require manual feature engineering, CNNs automatically learn spatial hierarchies of features (e.g., color transitions from green to yellow, dark spots, bruises) directly from image pixels, making them ideal for high-accuracy image classification.

### Computer Vision & Preprocessing
Raw input images vary in size, lighting, and background noise. Computer Vision techniques (OpenCV) standardize input images ($224 \times 224$ pixels), normalize pixel values, and enhance image contrast. This improves the generalization performance and accuracy of the neural network.

### Rule-Based Expert System
While Neural Networks excel at classification, they act as "black boxes" that do not directly generate market recommendations. Integrating a Rule-Based System allows the model output (quality grade & confidence score) to be processed through explainable IF-THEN expert rules to determine actionable pricing and storage advice for vendors.

---

## 7. Expected Inputs and Outputs

### Inputs
* **Mango Image**: Uploaded photo (JPEG/PNG) of a mango.
* **Base Market Price (Optional)**: Benchmark market price per kg for standard Grade A mangoes.

### Processing Pipeline
1. Image resizing ($224 \times 224$) and pixel normalization via OpenCV.
2. Feature extraction and class probability prediction via trained CNN.
3. Logical evaluation of quality grade via Rule-Based Expert Engine.

### Outputs
* **Predicted Fruit Type & Class**: Mango (*Grade A - Ripe*, *Grade B - Unripe*, *Grade C - Overripe/Damaged*).
* **Freshness / Ripeness Confidence Score**: Percentage probability (e.g., 94.2% Ripe).
* **Estimated Shelf Life**: Estimated remaining days before spoilage (e.g., Grade A = 3-5 days, Grade B = 7-10 days, Grade C = 1 day).
* **Recommended Selling Price**: Suggested retail price per kg based on grade discount rules.

---

## 8. Proposed Tools and Technologies (100% Free & Open-Source Stack)

| Component | Technology / Tool | Usage & Description |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.x | Core programming language for AI development. |
| **AI / Deep Learning** | TensorFlow / Keras | Training the CNN model architecture. |
| **Computer Vision** | OpenCV | Image preprocessing, resizing, and normalization. |
| **Model Training Hardware** | Google Colab | Free cloud T4 GPU hardware acceleration. |
| **Dataset Source** | Kaggle Datasets | Free Mango Ripeness & Quality Image Datasets. |
| **Frontend / Web UI** | Streamlit | Lightweight, open-source Python web framework. |
| **IDE & Version Control** | VS Code & GitHub | Code editor and project repository management. |

---

## 9. Initial System Workflow

The system executes a 5-step automated workflow:

```
[ Step 1: Image Upload ]
       ⬇
[ Step 2: OpenCV Preprocessing (224x224, Normalization) ]
       ⬇
[ Step 3: CNN Classification (Grade A / Grade B / Grade C) ]
       ⬇
[ Step 4: Rule-Based Engine (Price & Shelf-Life Rules) ]
       ⬇
[ Step 5: Output Display (Grade, Confidence %, Price, Shelf Life) ]
```

1. **Image Input**: User uploads a mango image through the Streamlit web interface.
2. **Preprocessing**: OpenCV standardizes the image resolution to $224 \times 224$ pixels and normalizes RGB intensity values.
3. **CNN Evaluation**: The trained CNN extracts visual color/texture features and outputs class probability distribution.
4. **Rule Engine Execution**: The predicted class is fed into the decision engine to calculate price adjustments and remaining shelf life.
5. **Result Presentation**: Streamlit interface displays the output breakdown clearly on screen.

---

## 10. Work Division

| Group Member | Core Responsibilities |
| :--- | :--- |
| **Member 1 (Technical & AI Lead)** | Designing CNN neural network architecture, model training in TensorFlow/Keras on Google Colab GPU, hyperparameter optimization, and accuracy evaluation. |
| **Member 2 (Computer Vision & Data)** | Dataset acquisition from Kaggle, data cleaning and augmentation (rotation/flip), OpenCV preprocessing pipeline development. |
| **Member 3 (Rule Engine & Frontend)** | Developing the Rule-Based decision logic for price/shelf-life recommendations, building the Streamlit UI, and integrating the trained model (`.h5` file). |
| **Member 4 (Testing & Documentation)** | Project report drafting, system integration testing with test image samples, ethical & social impact analysis, and presentation slides creation. |

---

## 11. Timeline (10-Week Plan)

* **Weeks 1 – 2**: Problem identification, scope minimization to single fruit (Mango), literature review, and lecturer approval.
* **Weeks 3 – 4**: Dataset collection from Kaggle, defining system workflow, and **Stage 1 Proposal Submission**.
* **Weeks 5 – 6**: Image preprocessing, CNN architecture implementation, and model training/validation on Google Colab.
* **Weeks 7 – 8**: Rule-based decision engine integration, Streamlit Web UI development, and **Stage 2 Progress Review**.
* **Week 9**: End-to-end system testing, performance metric evaluation (Confusion Matrix, Precision, Recall), UI refining.
* **Week 10**: Finalizing complete project report, recording presentation demo video, preparing slides, and **Stage 3 Final Submission**.

---

## 12. References

1. Gill, H. S., & Kular, R. S. (2022). *Fruit Quality Grading and Classification Using Deep Learning and Convolutional Neural Networks*. Journal of Agricultural AI & Automation, 14(3), 112–124.
2. Mamat, N., et al. (2020). *Advanced Computer Vision and Artificial Neural Network for Automatic Fruit Freshness Detection*. IEEE Access, 8, 148560–148572.
3. Kaggle Datasets. (2023). *Mango Ripeness & Quality Dataset for Computer Vision*. Retrieved from https://www.kaggle.com
4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
