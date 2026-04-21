# ML Disease Predictor - Healthcare Decision Support System

![Healthcare ML](https://cdn.activestate.com/wp-content/uploads/2018/10/machine-learning-healthcare-blog-hero-1200x799.jpg)

## Overview

**ML Disease Predictor** is a comprehensive web-based healthcare decision support application that leverages machine learning and deep learning models to predict the presence of multiple diseases. The system integrates both traditional machine learning classifiers (for tabular data) and deep neural networks (for medical imaging) to provide non-invasive preliminary disease risk assessments.

**Disclaimer:** This application is intended for educational and research purposes only. Predictions should not be used as a substitute for professional medical diagnosis or treatment.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Supported Diseases](#supported-diseases)
- [Technical Stack](#technical-stack)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Model Details](#model-details)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)

---

## Features

✅ **Multi-Disease Support** - Simultaneous detection of 7 different health conditions  
✅ **Dual ML Approach** - Combines traditional ML for clinical parameters and Deep Learning for medical imaging  
✅ **Image Classification** - CNN-based detection for Malaria and Pneumonia from microscopy/X-ray images  
✅ **Clinical Parameter Analysis** - Logistic Regression models for structured medical data  
✅ **Confidence Scoring** - Real-time probability estimates for prediction confidence  
✅ **User-Friendly Web Interface** - Responsive Flask-based web application  
✅ **Fast Inference** - Pre-trained models for immediate predictions  

---

## Architecture

The application follows a modular architecture with clear separation of concerns:

```
Input Data
    ↓
[Image Upload] ──→ CNN Models (Malaria/Pneumonia)
                     ↓
[Clinical Form] ──→ Logistic Regression (Other Diseases)
                     ↓
[Prediction Engine] ──→ Confidence Scoring ──→ Results Display
```

**Key Components:**
- **Flask Web Server** - REST API and web interface
- **Pre-trained Models** - 7 serialized ML models
- **Model Loading Layer** - TensorFlow/Keras for deep learning, Scikit-learn for classical ML
- **Image Processing** - TensorFlow/Pillow for medical image preprocessing
- **Frontend** - HTML/CSS templates with form validation

---

## Supported Diseases

| Disease | Model Type | Input Type | Reference |
|---------|-----------|-----------|-----------|
| **Cancer (Breast)** | Logistic Regression | Clinical Parameters (30 features) | Wisconsin Dataset |
| **Diabetes** | Logistic Regression | Clinical Parameters (8 features) | Pima Indians Dataset |
| **Heart Disease** | Logistic Regression | Clinical Parameters (11 features) | Cleveland Dataset |
| **Liver Disease** | Logistic Regression | Clinical Parameters (10 features) | Indian Liver Patient Records |
| **Kidney Disease (CKD)** | Logistic Regression | Clinical Parameters (12 features) | CKD Dataset |
| **Malaria** | Deep CNN | Cell Images (50×50×3 px) | Cell Images for Malaria Detection |
| **Pneumonia** | Deep CNN | Chest X-rays (64×64×3 px) | Chest X-ray Dataset |

---

## Technical Stack

### Backend
- **Python 3.x** - Core language
- **Flask 1.0.2** - Web framework and routing
- **TensorFlow 2.5.0** - Deep learning framework
- **Scikit-learn 0.22.1** - Machine learning algorithms
- **Numpy 1.15.1** - Numerical computing
- **Pandas 0.25.1** - Data manipulation
- **Pillow 8.3.2** - Image processing
- **Gunicorn 19.9.0** - Production WSGI server

### Frontend
- **HTML5** - Markup structure
- **CSS3** - Styling and responsive design
- **Jinja2 2.10.1** - Template engine

### Deployment
- **Heroku/Cloud** - Via Procfile configuration

---

## Installation & Setup

### Prerequisites
- Python 3.6 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd MLDiseasepredictor
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Files
Ensure all pre-trained models are present in the root directory:
```
model          # Cancer model
model1         # Diabetes model
model2         # Heart model
model3         # Kidney model
model4         # Liver model
model111.h5    # Malaria model
my_model.h5    # Pneumonia model
```

### Step 5: Create Required Directories
```bash
mkdir uploads
```

### Step 6: Run Application
```bash
# Development
python app.py

# Production (Gunicorn)
gunicorn app:app
```

The application will be available at `http://localhost:5000`

---

## Usage Guide

### For Tabular Data Predictions (Cancer, Diabetes, Heart, Liver, Kidney)

1. Navigate to the disease-specific page (e.g., `/diabetes`)
2. Enter the required clinical parameters in the form
3. Click "Predict" button
4. View results with confidence scoring

**Parameters Required by Disease:**

- **Diabetes** (8 parameters): Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree, Age
- **Cancer** (30 parameters): Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, etc.
- **Heart** (11 parameters): Age, Sex, Chest Pain Type, Resting BP, Cholesterol, etc.
- **Liver** (10 parameters): Age, Gender, Albumin, Bilirubin, etc.
- **Kidney** (12 parameters): Age, Blood Pressure, Glucose, Blood Urea, etc.

### For Image-Based Predictions (Malaria, Pneumonia)

1. Navigate to `/Malaria` or `/Pneumonia` page
2. Upload a medical image (cell microscopy or X-ray)
3. System automatically preprocesses and classifies the image
4. Displays prediction with accuracy percentage

---

## Model Details

### Deep Learning Models (CNN)

**Malaria Detection Model (`model111.h5`)**
- Input: 50×50×3 RGB cell images
- Architecture: Convolutional Neural Network with multiple layers
- Output Classes: [Parasitic, Uninfected, Invasive Carcinoma, Normal]
- Preprocessing: Image normalization (0-1 scaling)

**Pneumonia Detection Model (`my_model.h5`)**
- Input: 64×64×3 RGB X-ray images
- Architecture: Deep CNN with pooling layers
- Output Classes: [Normal, Pneumonia]
- Preprocessing: Image normalization (0-1 scaling)

### Classical ML Models (Logistic Regression)

All remaining disease models use **Logistic Regression** with the following characteristics:
- Algorithm: L2-regularized logistic regression
- Feature Scaling: StandardScaler applied (for applicable datasets)
- Cross-Validation: 10-fold CV for model evaluation
- Class Imbalance Handling: SMOTE applied (for Kidney and Liver datasets)

---

## Dataset Information

### Local Datasets (in repository)
- `cancer.csv` - Breast cancer features and diagnosis labels
- `diabetes.csv` - Pima Indians diabetes screening data
- `heart.csv` - Cleveland heart disease dataset

### External Datasets (Kaggle)

| Disease | Source | Records | Features |
|---------|--------|---------|----------|
| Liver | [Indian Liver Patient](https://www.kaggle.com/uciml/indian-liver-patient-records) | 583 | 10 |
| Kidney | [CKD Disease](https://www.kaggle.com/mansoordaku/ckdisease) | 400 | 24 |
| Malaria | [Cell Images](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) | 27,558 images | - |
| Pneumonia | [Chest X-ray](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) | 5,856 images | - |

### Data Preprocessing Pipeline

1. **Missing Value Handling**: Forward-fill method for time-series, median imputation for others
2. **Categorical Encoding**: One-hot encoding for multi-class features, binary mapping for binary features
3. **Feature Scaling**: StandardScaler normalization (mean=0, std=1)
4. **Class Balancing**: SMOTE for imbalanced datasets (Kidney, Liver)
5. **Feature Selection**: Irrelevant features dropped based on domain knowledge

---

## Project Structure

```
MLDiseasepredictor/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku deployment config
│
├── Training Scripts/
│   ├── diabetes.py                # Diabetes model training
│   ├── cancer.py                  # Cancer model training
│   ├── heart.py                   # Heart model training
│   ├── kidney.py                  # Kidney model training
│   ├── liver.py                   # Liver model training
│   └── alzh.py                    # Alzheimer's model training
│
├── Pre-trained Models/
│   ├── model                       # Cancer (Logistic Regression)
│   ├── model1                      # Diabetes (Logistic Regression)
│   ├── model2                      # Heart (Logistic Regression)
│   ├── model3                      # Kidney (Logistic Regression)
│   ├── model4                      # Liver (Logistic Regression)
│   ├── model111.h5                # Malaria (Deep CNN)
│   └── my_model.h5                # Pneumonia (Deep CNN)
│
├── Datasets/
│   ├── cancer.csv
│   ├── diabetes.csv
│   └── heart.csv
│
├── static/
│   └── main.css                   # Frontend styling
│
├── template/
│   ├── home.html                  # Landing page
│   ├── about.html                 # About page
│   ├── diabetes.html              # Diabetes form
│   ├── cancer.html                # Cancer form
│   ├── heart.html                 # Heart form
│   ├── liver.html                 # Liver form
│   ├── kidney.html                # Kidney form
│   ├── index.html                 # Malaria upload page
│   ├── index2.html                # Pneumonia upload page
│   ├── predict.html               # Image prediction results
│   ├── predict1.html              # Image prediction results (v2)
│   ├── result.html                # Tabular prediction results
│   └── login.html                 # Authentication (optional)
│
└── uploads/                       # User-uploaded images (runtime)
```

---

## API Endpoints

### Web Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home page |
| `/home` | GET | Home page |
| `/about` | GET | About page |
| `/cancer` | GET | Cancer prediction form |
| `/diabetes` | GET | Diabetes prediction form |
| `/heart` | GET | Heart disease prediction form |
| `/liver` | GET | Liver disease prediction form |
| `/kidney` | GET | Kidney disease prediction form |
| `/Malaria` | GET | Malaria image upload |
| `/Pneumonia` | GET | Pneumonia image upload |

### Prediction Routes

| Route | Method | Purpose | Input |
|-------|--------|---------|-------|
| `/result` | POST | Process tabular predictions | Form data (8-30 fields) |
| `/upload` | POST/GET | Malaria image prediction | Image file |
| `/upload11` | POST/GET | Pneumonia image prediction | Image file |
| `/uploads/<filename>` | GET | Serve uploaded files | Filename |

---

## Contributing

### Development Guidelines

1. **Code Quality**: Follow PEP 8 standards
2. **Testing**: Test all models before deployment
3. **Documentation**: Update README for new features
4. **Model Updates**: Retrain models with fresh data quarterly

### Future Enhancements

- [ ] Add more disease prediction models
- [ ] Implement user authentication system
- [ ] Add prediction history tracking
- [ ] Deploy REST API for mobile integration
- [ ] Implement explainable AI (SHAP/LIME) for model interpretability
- [ ] Add ensemble methods for improved accuracy
- [ ] Create data visualization dashboards
- [ ] Implement automated model retraining pipeline

---

## Contact & References

**Developer**: [Musonda Salimu](https://www.linkedin.com/in/musonda-salimu-a4a0b31b9/)

### References & Acknowledgments

- Kaggle Kernels: Machine Learning tutorials and datasets
- TensorFlow/Keras Documentation
- Scikit-learn community resources
- UCI Machine Learning Repository

---

## License

This project is provided for educational and research purposes.

---

**Last Updated**: April 2026  
**Version**: 1.0  
**Status**: Production Ready
</ul>
<hr>
 <h3> To run this project, clone the repository and type the following commands in the termial: </h3>
 <ul>
  <li> $ set FLASK_APP= app.py</li>
  <li> $ flask run</li>
  </ul>
  <hr>
  <p> Thank you!</p>
