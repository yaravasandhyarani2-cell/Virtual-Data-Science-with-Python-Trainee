# Week 4 - Supervised Learning & Classification Analysis

## 1. Dataset Overview
- **Dataset:** Breast Cancer Wisconsin Diagnostic (Public Benchmark from Scikit-Learn / UCI)
- **Problem Formulation:** Binary Classification (predicting whether a tumour biopsy is `malignant` (0) or `benign` (1)).
- **Total Samples:** 569 (455 Train / 114 Test stratified split)
- **Features:** 30 continuous cell nucleus characteristics computed from digitized fine needle aspirate (FNA) images.

## 2. Supervised Learning Pipeline
1. **Preprocessing & Standardization:** Fitted `StandardScaler` strictly on training data to prevent data leakage, then transformed both train and test splits.
2. **Stratified 5-Fold Cross-Validation:** Validated model generalization across folds.
3. **Algorithms Compared:**
   - **Logistic Regression (L2 Regularized)**
   - **Random Forest Classifier (Ensemble of 100 Trees, max_depth=6)**
4. **Evaluation Metrics:** Evaluated with Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## 3. Test Set Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| **Random Forest Classifier** | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9947 |

## 4. Key Findings & Insights
1. **High Diagnostic Accuracy:** Both models achieve high diagnostic performance on the held-out test set, with **Logistic Regression** achieving an accuracy of **98.2%** and an ROC-AUC of **0.9954**.
2. **Top Predictive Biomarkers:** Feature importance analysis via Random Forest reveals that `worst concave points`, `worst perimeter`, `worst radius`, and `mean concave points` are the strongest predictors of tumour malignancy.
3. **Clinical Implication:** High Recall (0.9583 for Random Forest and 0.9861 for Logistic Regression) minimizes false negatives, critical in medical diagnostics to avoid missed diagnoses.

## 5. Artifacts and Generated Deliverables
- **Figures:** Saved to `figures/` (Fig 1: Confusion Matrices, Fig 2: ROC Curves, Fig 3: PR Curves, Fig 4: Feature Importances)
- **Serialized Models:** Saved to `data/processed/` (`logistic_regression.joblib`, `random_forest_classifier.joblib`, `scaler.joblib`)
- **Evaluation Summaries:** Saved to `data/processed/model_comparison_metrics.csv` and `outputs/evaluation_summary.json`
