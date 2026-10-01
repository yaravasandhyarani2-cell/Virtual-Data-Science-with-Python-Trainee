# Week 5 - Deep Learning Application: Neural Network Classification

## 1. Project Overview
- **Framework:** TensorFlow 2.21.0 / Keras
- **Dataset:** Breast Cancer Wisconsin Diagnostic (Public Benchmark via Scikit-Learn)
- **Objective:** Design, compile, train, and evaluate a Deep Feedforward Neural Network to classify tumour biopsies as **Malignant** (0) or **Benign** (1).
- **Dataset Dimensions:** 569 records, 30 continuous input features.

## 2. Neural Network Architecture
The sequential architecture was engineered with deep regularization blocks to prevent overfitting on small-to-medium clinical tabular data:
- **Input Layer:** 30 normalized clinical features
- **Hidden Layer 1:** 64 units, ReLU activation, L2 weight regularization (1e-4) + BatchNormalization + Dropout (30%)
- **Hidden Layer 2:** 32 units, ReLU activation, L2 weight regularization (1e-4) + BatchNormalization + Dropout (20%)
- **Hidden Layer 3:** 16 units, ReLU activation + Dropout (10%)
- **Output Layer:** 1 unit, Sigmoid activation (calibrated class probability)

## 3. Training Strategy & Overfitting Prevention
- **Optimization:** Adam optimizer with initial learning rate 0.001
- **Loss Function:** Binary Cross-Entropy
- **Callbacks Implemented:**
  1. `EarlyStopping`: Monitored `val_loss` with patience of 15 epochs and restored best model weights.
  2. `ReduceLROnPlateau`: Halved learning rate when validation loss plateaued for 7 epochs.
  3. `ModelCheckpoint`: Persisted best model weights checkpoint to `models/best_model_weights.weights.h5`.

## 4. Evaluation Results on Held-Out Test Set
- **Test Accuracy:** 0.9651 (96.5%)
- **Test Precision:** 0.9636
- **Test Recall:** 0.9815
- **Test F1-Score:** 0.9725
- **Test ROC-AUC:** 0.9925
- **Epochs Trained:** 54 (Early Stopping converged smoothly)

## 5. Artifacts and Generated Deliverables
- **Models:** Saved to `models/` (`deep_classifier_model.keras` and `best_model_weights.weights.h5`)
- **Figures:** Saved to `figures/` (Fig 1: Training History Curves, Fig 2: Confusion Matrix, Fig 3: ROC Curve)
- **Metrics Summary:** Saved to `outputs/deep_learning_summary.json` and `data/processed/deep_learning_metrics.csv`
