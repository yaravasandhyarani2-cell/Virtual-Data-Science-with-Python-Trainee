# Week 5: Deep Learning Application with TensorFlow / Keras

## Overview
This module implements an end-to-end, modular **Deep Learning Application** using **TensorFlow 2.21 & Keras** for binary classification on the benchmark **Breast Cancer Wisconsin Diagnostic** dataset.

The system designs, trains, regularizes, and evaluates a multi-layer deep neural network with automated early stopping, learning rate adaptation, and weight checkpointing.

---

## Directory Structure
```text
Week5/
├── data/
│   ├── raw/
│   │   └── raw.csv                          # Untouched raw dataset (569 rows x 32 columns)
│   └── processed/
│       ├── train_scaled.csv                 # Normalized training set (70% split, 398 samples)
│       ├── val_scaled.csv                   # Normalized validation set (15% split, 85 samples)
│       ├── test_scaled.csv                  # Normalized test set (15% split, 86 samples)
│       ├── deep_learning_metrics.csv        # Final evaluation metrics on held-out test set
│       └── deep_learning_summary.json       # JSON evaluation summary
├── figures/
│   ├── fig01_training_history_curves.png    # Training vs Validation loss & accuracy curves
│   ├── fig02_test_confusion_matrix.png      # Held-out test set confusion matrix
│   └── fig03_test_roc_curve.png             # Test set ROC curve with AUC metric
├── models/
│   ├── best_model_weights.weights.h5        # Best weights checkpoint saved via ModelCheckpoint
│   └── deep_classifier_model.keras          # Full serialized Keras deep learning model
├── outputs/
│   ├── deep_learning_summary.json           # Machine-readable architecture & metric summary
│   └── key_findings.md                      # Detailed findings, model metrics & training analysis
├── src/
│   └── deep_learning_model.py               # Complete modular, executable deep learning script
└── requirements.txt                         # Package dependencies
```

---

## Deep Learning Architecture & Pipeline

### 1. Model Architecture
A deep sequential feedforward neural network engineered with multi-level regularization:
- **Input Dimension:** 30 normalized continuous clinical nucleus features
- **Dense Layer 1:** 64 units, ReLU activation, $L_2$ weight regularization ($10^{-4}$)
- **Batch Normalization 1:** Stabilizes activation distributions and gradient flow
- **Dropout 1:** 30% dropout rate for co-adaptation prevention
- **Dense Layer 2:** 32 units, ReLU activation, $L_2$ weight regularization ($10^{-4}$)
- **Batch Normalization 2:** Second normalization stage
- **Dropout 2:** 20% dropout rate
- **Dense Layer 3:** 16 units, ReLU activation
- **Dropout 3:** 10% dropout rate
- **Output Layer:** 1 unit, Sigmoid activation (calibrated class probability)

### 2. Compilation
- **Optimizer:** Adam (initial learning rate = $0.001$)
- **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
- **Metrics Tracked:** Accuracy, Precision, Recall, AUC

### 3. Training Callbacks
- **`EarlyStopping`:** Monitored `val_loss` with a patience of 15 epochs; automatically stopped training at epoch 54 and restored best weights from epoch 39.
- **`ReduceLROnPlateau`:** Dynamically halved learning rate when validation loss plateaued for 7 epochs.
- **`ModelCheckpoint`:** Persisted optimal weights to `models/best_model_weights.weights.h5`.

---

## Held-Out Test Set Performance (86 Samples)

| Metric | Score | Clinical Relevance |
|---|---|---|
| **Test Accuracy** | **96.51%** | High overall diagnostic accuracy |
| **Test Precision** | **96.36%** | Low false positive rate for benign tumours |
| **Test Recall** | **98.15%** | Minimizes critical false negatives (malignant biopsies missed) |
| **Test F1-Score** | **97.25%** | Balanced harmonic precision-recall measure |
| **Test ROC-AUC** | **0.9925** | Near-perfect discriminative capability |

---

## How to Run the Pipeline

From the project root:
```powershell
python Week5/src/deep_learning_model.py
```

Or from inside `Week5/`:
```powershell
cd Week5
python src/deep_learning_model.py
```
All training curves, confusion matrices, ROC plots, models, and metric summaries will be regenerated automatically.
