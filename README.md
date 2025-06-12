# 🏥 Multimodal Readmission Prediction

A deep learning project to predict 30-day hospital readmission using both **structured EHR data** and **unstructured clinical notes** from the **MIMIC-III Clinical Database**. This work explores and compares multiple multimodal fusion strategies using PyTorch, including early fusion, attention-based models, deep residual architectures, and late ensemble fusion.


## 📦 Dataset

- **Source**: [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/)
- **Components Used**:
  - `ADMISSIONS.csv`
  - `PATIENTS.csv`
  - `CHARTEVENTS.csv`
  - `LABEVENTS.csv`
  - `NOTEEVENTS.csv`

To access the dataset, you must complete the PhysioNet credentialing process.

## 🧠 Modules Overview

### 📘 Module 1 – Data Processing and Feature Engineering
- Extracts demographic and vital sign trends  
- Processes discharge summaries using ClinicalBERT  
- Saves structured features and text embeddings  

### 📗 Module 2 – Baseline Modeling
- Generates 30-day readmission labels  
- Trains XGBoost on structured data  
- Trains BERT classifier on text embeddings  
- Evaluates with AUC and classification metrics  

### 🔀 Module 3 – Multimodal Fusion
Combines structured and textual inputs in multiple ways:

- **3.1**: Deep MLP Fusion Model  
- **3.2**: Focal Loss + Threshold Optimization  
- **3.3**: DeepResidualFusion Model with skip connections  
- **3.4**: Late Fusion (Ensemble of Structured + Text Models)  

## 🔧 Model Architectures

- **Early Fusion**: Concatenation of structured + text embeddings → MLP  
- **Attention-Based Fusion**: Attention layers across modalities  
- **DeepResidualFusion**: Deep skip-connected fusion model  
- **Late Fusion**: Ensemble probabilities from separate structured and text models  


## 📊 Evaluation Metrics

- ROC AUC Score  
- Precision / Recall / F1  
- PR Curve & ROC Curve Visualizations  
- Uses stratified train/test splits to ensure fair evaluation  


## 📈 Results (Sample)

| Model                  | AUC Score |
|------------------------|-----------|
| XGBoost (structured)   | ~0.67     |
| BERT Classifier (text) | ~0.64     |
| Early Fusion MLP       | ~0.66     |
| DeepResidualFusion     | ~0.67     |
| Late Fusion (ensemble) | ~0.68     |


> ⚠️ **Note**: AUC may vary depending on data splits, feature quality, and text preprocessing.

### 🔍 Why the AUC Score is Low in This Prediction Model?

In this prediction model for 30-day hospital readmission, the low AUC score can be primarily attributed to **severe class imbalance**, where a small minority of patients are actually readmitted compared to those who are not. This imbalance leads the model to favor the majority class, making it less sensitive to identifying true positives and ultimately lowering its discriminatory power, as reflected in the AUC score.





