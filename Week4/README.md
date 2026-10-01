# Week 4: Supervised Learning and Classification Analysis

## Overview
This module implements an end-to-end, modular machine learning pipeline for **Supervised Binary Classification** using the public **Breast Cancer Wisconsin Diagnostic Dataset** (via Scikit-Learn / UCI Machine Learning Repository).

The goal is to accurately distinguish between **Malignant** (tumour) and **Benign** biopsies based on 30 continuous clinical cell nucleus morphological features.

---

## Directory Structure
```text
Week4/
├── data/
│   ├── raw/
│   │   └── raw.csv                          # Untouched raw dataset with 569 instances & 32 columns
│   └── processed/
│       ├── train_processed.csv              # Scaled training dataset (80% split)
│       ├── test_processed.csv               # Scaled test dataset (20% split)
│       ├── scaler.joblib                    # Serialized StandardScaler fitted on train
│       ├── logistic_regression.joblib       # Serialized trained Logistic Regression model
│       ├── random_forest_classifier.joblib  # Serialized trained Random Forest model
│       ├── model_comparison_metrics.csv     # Model evaluation comparison table
│       ├── feature_importances.csv          # Ranked Gini feature importances
│       └── evaluation_summary.json          # Machine-readable evaluation summary
├── figures/
│   ├── fig01_confusion_matrices.png         # Test set confusion matrices for both models
│   ├── fig02_roc_curves.png                 # Receiver Operating Characteristic (ROC) curves & AUC
│   ├── fig03_precision_recall_curves.png    # Precision-Recall curves & Average Precision
│   └── fig04_feature_importances.png        # Top 15 Random Forest feature importances
├── outputs/
│   ├── evaluation_summary.json              # Evaluation summary with CV and test metrics
│   └── key_findings.md                      # Comprehensive findings and clinical insights
├── src/
│   └── supervised_model.py                  # Modular, runnable supervised learning pipeline
└── requirements.txt                         # Python package dependencies
```

---

## Methodology & Machine Learning Workflow

1. **Data Acquisition**:
   - Loaded Breast Cancer Wisconsin Diagnostic benchmark dataset (569 rows, 30 numeric feature dimensions).
   - Target classes: `malignant` (212 samples, 37.3%) and `benign` (357 samples, 62.7%).

2. **Preprocessing & Standardization**:
   - Partitioned data using an 80/20 **Stratified Train-Test Split** (`random_state=42`) preserving class ratios.
   - Applied `StandardScaler` fitted strictly on training data to prevent data leakage and transformed both splits.

3. **Models & Cross-Validation**:
   - Compared **Logistic Regression** (L2 penalty) and **Random Forest Classifier** (100 estimators, max_depth=6).
   - Evaluated across 5 stratified folds using 5 metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

4. **Performance Evaluation on Held-Out Test Set**:

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9954** |
| **Random Forest Classifier** | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9947 |

5. **Key Biomarkers Identified**:
   - Top predictive features ranked by Gini Importance:
     1. `worst concave points`
     2. `worst perimeter`
     3. `worst radius`
     4. `mean concave points`
     5. `worst area`

---

## How to Run the Pipeline

From the repository root:
```powershell
python Week4/src/supervised_model.py
```

Or from inside the `Week4` folder:
```powershell
cd Week4
python src/supervised_model.py
```
All outputs, models, CSV summaries, and figures will be regenerated automatically.
