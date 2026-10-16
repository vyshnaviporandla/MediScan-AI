# MediScan AI 🩺
### AI-Powered Medical Image Analysis Using Deep Learning

MediScan AI is an educational artificial intelligence project that classifies chest X-ray images into two dataset categories: **NORMAL** and **PNEUMONIA**. Built with TensorFlow, Keras, MobileNetV2, and Streamlit, it demonstrates an end-to-end machine-learning workflow with an interactive web dashboard.

> ⚠️ **Medical Disclaimer:** MediScan AI is an educational prototype, not a clinically validated medical device. It must not be used to diagnose, rule out disease, or make treatment decisions. Always consult qualified healthcare professionals for medical concerns.

---

## 📌 Project Overview

Medical image analysis is an important research application of artificial intelligence. This project demonstrates how transfer learning can be used for chest X-ray image classification and how model outputs can be presented through a user-friendly web interface.

### Objectives

- Explore deep learning for medical image classification.
- Implement transfer learning with MobileNetV2.
- Build an interactive Streamlit dashboard.
- Visualize classification outputs and training performance.
- Evaluate a trained model using independent test data.
- Demonstrate responsible and transparent AI development.

## ✨ Key Features

- **Chest X-Ray Upload:** Upload an image through the web interface.
- **AI Classification:** Generate an output for NORMAL or PNEUMONIA.
- **Prediction Visualization:** Display model output and associated scores.
- **Confidence Chart:** Visualize the scores produced by the model.
- **Session History:** Review predictions from the current session.
- **Model Information:** Display information about the model configuration.
- **Training Curves:** Visualize model training performance.
- **Evaluation Metrics:** Measure accuracy, recall, specificity, and ROC-AUC.
- **Educational Interface:** Explore the machine-learning workflow in an interactive dashboard.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning and model training |
| MobileNetV2 | Transfer-learning architecture |
| Streamlit | Interactive web dashboard |
| NumPy | Numerical operations |
| Scikit-learn | Classification metrics and evaluation |
| Matplotlib | Training visualization |
| Git and GitHub | Version control and project hosting |

## 🧠 Model Architecture

MediScan AI uses MobileNetV2 pretrained on ImageNet as a feature extractor.

**Workflow:**

1. Load a chest X-ray image.
2. Resize the image to 224 × 224 pixels.
3. Apply training-time augmentation using random rotation and zoom.
4. Normalize the image using MobileNetV2 preprocessing.
5. Extract features using the frozen MobileNetV2 base.
6. Apply global average pooling and dropout.
7. Generate a binary classification score using a sigmoid output layer.

The model is compiled using the Adam optimizer and binary cross-entropy loss. Accuracy and area under the ROC curve (AUC) are tracked during training.

A model score is not necessarily a calibrated probability and should not be interpreted as a clinical risk estimate.

## 📊 Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset, commonly distributed through Kaggle.

The dataset contains images organized into NORMAL and PNEUMONIA categories. The local dataset was checked after preparing a larger validation split.

| Dataset split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,207 | 3,487 | 4,694 |
| Validation | 142 | 396 | 538 |
| Testing | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

The validation images were moved from the original training set to create a larger validation split. The test set was left unchanged.

Dataset counts can differ between versions. Check dataset licensing and terms before redistributing images. Do not upload the complete dataset to a public repository unless redistribution is explicitly permitted.

## 📈 Model Evaluation — Actual Test Results

The trained model was evaluated on **624 held-out test images**.

| Metric | Result |
|---|---:|
| Test accuracy | 78.69% |
| ROC-AUC | 0.9569 |
| PNEUMONIA recall / sensitivity | 99.74% |
| NORMAL recall / specificity | 43.59% |
| Test images | 624 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.99 | 0.44 | 0.61 | 234 |
| PNEUMONIA | 0.75 | 1.00 | 0.85 | 390 |
| **Overall accuracy** | | | **0.79** | **624** |

### Interpretation

- The model identified nearly all test images labelled PNEUMONIA at the selected threshold of 0.5.
- Specificity was low: only about 43.59% of images labelled NORMAL were correctly classified.
- This means many NORMAL-labelled images were incorrectly classified as PNEUMONIA.
- The high ROC-AUC indicates good ranking performance on this dataset, but does not establish clinical usefulness or generalization to other populations and imaging settings.

These metrics describe this experiment only. They do not establish diagnostic accuracy or clinical safety.

## 📂 Project Structure

```text
MedicalAI/
│
├── app.py
├── app_original_backup.py
├── main.py
├── requirements.txt
├── prepare_validation.py
├── extract_dataset.py
├── README.md
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── plotting.py
│
├── data/
│   └── raw/
│       └── chest_xray/
│           ├── train/
│           │   ├── NORMAL/
│           │   └── PNEUMONIA/
│           ├── val/
│           │   ├── NORMAL/
│           │   └── PNEUMONIA/
│           └── test/
│               ├── NORMAL/
│               └── PNEUMONIA/
│
├── models/
│   └── pneumonia_mobilenet.keras
│
└── outputs/
    ├── training_log.csv
    └── training_curves.png
```

Some files are generated only after training or evaluation. The dataset, virtual environment, and trained model should generally be kept out of GitHub unless repository storage and distribution permissions have been considered.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system.
- Git, if cloning from GitHub.
- A compatible TensorFlow environment.
- Sufficient memory and disk space.
- The dataset in the expected directory structure.

### 1. Clone the repository

Replace the placeholder with the actual repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd MedicalAI
```

Alternatively, download the source ZIP and extract it.

### 2. Create a virtual environment

**Windows Command Prompt:**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Arrange the image files as follows:

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 5. Train the model

Run this command from the project root:

```cmd
python main.py train --epochs 5
```

The training script saves the best checkpoint based on validation AUC to:

```text
models/pneumonia_mobilenet.keras
```

Training duration depends on available hardware and system configuration.

### 6. Evaluate the model

```cmd
python main.py evaluate
```

The evaluation command uses the test split and prints classification metrics.

### 7. Launch the dashboard

```cmd
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, commonly:

```text
http://localhost:8501
```

If the application reports that the model is unavailable, verify that the trained model exists at the expected path and that the app's model-loading configuration matches it.

## 🔐 Responsible AI and Limitations

- This project is for education and experimentation only.
- Predictions can be wrong, including for images similar to those in the training set.
- Dataset imbalance and image-source differences may affect results.
- The model has low specificity on the reported test set.
- The dataset's labels do not establish clinical diagnostic accuracy.
- External validation and clinical oversight would be necessary before any real-world medical application.
- Do not upload identifiable patient images or private medical information to public repositories.

## 🔮 Future Enhancements

- Improve specificity and investigate false-positive classifications.
- Compare multiple model architectures and training strategies.
- Add confusion-matrix and ROC-curve visualizations.
- Improve reproducibility and experiment tracking.
- Add unit tests and automated validation checks.
- Enhance dashboard accessibility and responsive design.
- Explore explainability methods for educational analysis.
- Evaluate performance on additional authorized datasets.

Explainability visualizations would be exploratory and would not prove that a prediction is medically correct.

## 🎓 Learning Outcomes

This project demonstrates practical experience with:

- Python project development.
- Deep learning and transfer learning.
- TensorFlow/Keras model training.
- Image preprocessing and augmentation.
- Dataset splitting and evaluation.
- Classification metrics and visualizations.
- Streamlit dashboard development.
- Responsible AI principles.

## 👩‍💻 Project Information

**Project Name:** MediScan AI  
**Category:** Artificial Intelligence / Machine Learning  
**Domain:** Educational Medical Image Analysis  
**Model:** MobileNetV2  
**Interface:** Streamlit

**GitHub Repository:** Add your repository URL here.

**Developer Profile:** Add your GitHub profile URL here.

---

**MediScan AI — Exploring deep learning for educational chest X-ray classification with responsible AI practices.**
