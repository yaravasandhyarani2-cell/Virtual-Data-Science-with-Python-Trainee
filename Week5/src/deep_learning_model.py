"""
Week 5: Deep Learning Application with TensorFlow / Keras
Dataset: Breast Cancer Wisconsin Diagnostic (Public Benchmark from Scikit-Learn)
Author: Virtual Data Science with Python Internship

Modular Deep Learning Pipeline:
  1. Load and preprocess/normalize the dataset (StandardScaler + stratified train/val/test splits) -> data/raw/ & data/processed/
  2. Design a deep feedforward neural network (Sequential architecture with BatchNormalization, Dropout, and ReLU/Sigmoid)
  3. Compile model with Adam optimizer, Binary Cross-Entropy loss, and performance metrics
  4. Train model with validation split, Early Stopping, and ModelCheckpoint callbacks to prevent overfitting
  5. Evaluate performance using Accuracy, Loss curves, ROC-AUC, and Confusion Matrix
  6. Save training history plots to figures/ and trained model weights/architecture to models/
"""

import os
import json
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set working directory dynamically to Week5 root
script_dir = os.path.dirname(os.path.abspath(__file__))   # .../Week5/src
week5_root = os.path.dirname(script_dir)                  # .../Week5
os.chdir(week5_root)
print(f"[INFO] Working directory set to: {os.getcwd()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

# Global reproducibility
RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Plot styling
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

# Directory structure
RAW_PATH = os.path.join("data", "raw", "raw.csv")
PROC_DIR = os.path.join("data", "processed")
FIG_DIR = os.path.join("figures")
MODEL_DIR = os.path.join("models")
OUT_DIR = os.path.join("outputs")

for d in [os.path.dirname(RAW_PATH), PROC_DIR, FIG_DIR, MODEL_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

fig_counter = [0]

def save_fig(name: str) -> str:
    """Save plot to figures/ with sequential numbering."""
    fig_counter[0] += 1
    fname = f"fig{fig_counter[0]:02d}_{name}.png"
    fpath = os.path.join(FIG_DIR, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED FIGURE] {fpath}")
    return fpath


# =============================================================================
# 1. DATA ACQUISITION & PREPROCESSING / NORMALIZATION
# =============================================================================
def acquire_and_preprocess_data():
    """
    Load dataset, save raw copy, perform stratified 70/15/15 train/val/test split,
    normalize features with StandardScaler, and persist processed CSVs.
    """
    print("\n" + "="*60)
    print("SECTION 1 - DATA ACQUISITION & PREPROCESSING")
    print("="*60)

    if os.path.exists(RAW_PATH):
        print(f"[INFO] Raw data found at {RAW_PATH}. Loading...")
        df = pd.read_csv(RAW_PATH)
    else:
        print("[INFO] Fetching Breast Cancer dataset from sklearn...")
        bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame.copy()
        df["target_name"] = df["target"].map({0: "malignant", 1: "benign"})
        df.to_csv(RAW_PATH, index=False)
        print(f"[INFO] Raw data saved -> {RAW_PATH}")

    print(f"[INFO] Total Dataset Shape: {df.shape}")
    feature_cols = [c for c in df.columns if c not in ["target", "target_name"]]
    X = df[feature_cols].copy()
    y = df["target"].astype(int).copy()

    # Stratified Train-Temp Split (70% train, 30% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    # Split Temp into 50% Validation and 50% Test (15% val, 15% test of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"[INFO] Split sizes: Train={X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%), "
          f"Validation={X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%), "
          f"Test={X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Normalize features using StandardScaler fitted strictly on Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Persist processed splits
    pd.DataFrame(X_train_scaled, columns=feature_cols).assign(target=y_train.values).to_csv(
        os.path.join(PROC_DIR, "train_scaled.csv"), index=False
    )
    pd.DataFrame(X_val_scaled, columns=feature_cols).assign(target=y_val.values).to_csv(
        os.path.join(PROC_DIR, "val_scaled.csv"), index=False
    )
    pd.DataFrame(X_test_scaled, columns=feature_cols).assign(target=y_test.values).to_csv(
        os.path.join(PROC_DIR, "test_scaled.csv"), index=False
    )
    print(f"[INFO] Processed and scaled splits saved to {PROC_DIR}/")

    return (X_train_scaled, y_train.values,
            X_val_scaled, y_val.values,
            X_test_scaled, y_test.values,
            feature_cols, df.shape)


# =============================================================================
# 2. DEEP NEURAL NETWORK ARCHITECTURE
# =============================================================================
def build_model(input_dim: int) -> keras.Model:
    """
    Design a Deep Feedforward Sequential Neural Network with:
      - Dense input layer with 64 units and ReLU activation
      - BatchNormalization for training stability
      - Dropout (30%) for regularization to prevent overfitting
      - Dense hidden layer with 32 units and ReLU activation
      - BatchNormalization + Dropout (20%)
      - Dense hidden layer with 16 units and ReLU activation
      - Sigmoid output unit for binary classification probability
    """
    print("\n" + "="*60)
    print("SECTION 2 - NEURAL NETWORK ARCHITECTURE")
    print("="*60)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="input_features"),
        
        # Dense Block 1
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.30, name="dropout_1"),
        
        # Dense Block 2
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.20, name="dropout_2"),
        
        # Dense Block 3
        layers.Dense(16, activation="relu", name="dense_3"),
        layers.Dropout(0.10, name="dropout_3"),
        
        # Output Layer
        layers.Dense(1, activation="sigmoid", name="output_classification")
    ], name="BreastCancer_DeepClassifier")

    model.summary()
    return model


# =============================================================================
# 3. MODEL COMPILATION
# =============================================================================
def compile_model(model: keras.Model) -> keras.Model:
    """Compile model with Adam optimizer, binary cross-entropy loss, and metrics."""
    print("\n" + "="*60)
    print("SECTION 3 - MODEL COMPILATION")
    print("="*60)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.BinaryCrossentropy()
    metric_list = [
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metric_list)
    print("[INFO] Model compiled successfully with Adam optimizer (lr=0.001) & BinaryCrossentropy loss.")
    return model


# =============================================================================
# 4. TRAINING WITH CALLBACKS & EARLY STOPPING
# =============================================================================
def train_model(model: keras.Model, X_train, y_train, X_val, y_val):
    """
    Train model using EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.
    """
    print("\n" + "="*60)
    print("SECTION 4 - MODEL TRAINING & VALIDATION")
    print("="*60)

    best_weights_path = os.path.join(MODEL_DIR, "best_model_weights.weights.h5")
    full_model_path = os.path.join(MODEL_DIR, "deep_classifier_model.keras")

    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode="min"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-5,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=training_callbacks,
        verbose=1
    )

    # Save final full model
    model.save(full_model_path)
    print(f"\n[SAVED MODEL] Full model saved to: {full_model_path}")
    print(f"[SAVED WEIGHTS] Best weights checkpoint saved to: {best_weights_path}")

    return history, full_model_path, best_weights_path


# =============================================================================
# 5. EVALUATION AND VISUALIZATIONS
# =============================================================================
def evaluate_and_visualize(model, history, X_test, y_test):
    """
    Generate loss/accuracy learning curves, confusion matrix, ROC curve, and
    compute test evaluation metrics.
    """
    print("\n" + "="*60)
    print("SECTION 5 - PERFORMANCE EVALUATION & VISUALIZATIONS")
    print("="*60)

    # 1. Loss and Accuracy Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_trained = len(history.history["loss"])

    # Loss Curve
    axes[0].plot(range(1, epochs_trained + 1), history.history["loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(range(1, epochs_trained + 1), history.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_title("Training vs Validation Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend()

    # Accuracy Curve
    axes[1].plot(range(1, epochs_trained + 1), history.history["accuracy"], label="Train Accuracy", color="#2ca02c", lw=2)
    axes[1].plot(range(1, epochs_trained + 1), history.history["val_accuracy"], label="Val Accuracy", color="#d62728", lw=2, linestyle="--")
    axes[1].set_title("Training vs Validation Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle("Fig 1 - Neural Network Training & Validation Learning Curves", fontweight="bold", y=1.03)
    plt.tight_layout()
    save_fig("training_history_curves")

    # 2. Test Set Predictions
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n--- Held-Out Test Set Performance ---")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test Precision: {prec:.4f}")
    print(f"  Test Recall   : {rec:.4f}")
    print(f"  Test F1-Score : {f1:.4f}")
    print(f"  Test ROC-AUC  : {auc:.4f}")

    # 3. Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"])
    ax.set_title("Fig 2 - Test Set Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    save_fig("test_confusion_matrix")

    # 4. ROC Curve Visualization
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Deep Neural Net (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Fig 3 - Test Set Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig("test_roc_curve")

    metrics_dict = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "epochs_trained": epochs_trained,
        "final_train_loss": round(float(history.history["loss"][-1]), 4),
        "final_val_loss": round(float(history.history["val_loss"][-1]), 4),
        "final_train_acc": round(float(history.history["accuracy"][-1]), 4),
        "final_val_acc": round(float(history.history["val_accuracy"][-1]), 4),
    }

    return metrics_dict


# =============================================================================
# 6. PERSISTING OUTPUTS & SUMMARIES
# =============================================================================
def save_summaries(metrics_dict, raw_shape, feature_cols):
    """Save evaluation summary JSON and detailed key_findings.md."""
    print("\n" + "="*60)
    print("SECTION 6 - PERSISTING ARTIFACTS AND SUMMARIES")
    print("="*60)

    summary = {
        "dataset": "Breast Cancer Wisconsin Diagnostic",
        "task": "Deep Learning Binary Classification (Malignant vs Benign)",
        "framework": f"TensorFlow {tf.__version__} / Keras {keras.__version__}",
        "architecture": "Sequential Deep Neural Network (Dense 64 -> BN -> Drop(0.3) -> Dense 32 -> BN -> Drop(0.2) -> Dense 16 -> Drop(0.1) -> Sigmoid)",
        "optimizer": "Adam (lr=0.001)",
        "loss": "BinaryCrossentropy",
        "total_samples": int(raw_shape[0]),
        "features_count": len(feature_cols),
        "test_metrics": metrics_dict,
    }

    # Save JSON to both outputs/ and data/processed/
    for p in [os.path.join(OUT_DIR, "deep_learning_summary.json"),
              os.path.join(PROC_DIR, "deep_learning_summary.json")]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[SAVED] {p}")

    # Save metrics CSV
    metrics_df = pd.DataFrame([{
        "Model": "Deep Neural Network",
        "Accuracy": metrics_dict["accuracy"],
        "Precision": metrics_dict["precision"],
        "Recall": metrics_dict["recall"],
        "F1-Score": metrics_dict["f1"],
        "ROC-AUC": metrics_dict["roc_auc"],
        "Epochs_Trained": metrics_dict["epochs_trained"],
    }])
    csv_path = os.path.join(PROC_DIR, "deep_learning_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    # Generate Markdown Report
    md_content = f"""# Week 5 - Deep Learning Application: Neural Network Classification

## 1. Project Overview
- **Framework:** TensorFlow {tf.__version__} / Keras
- **Dataset:** Breast Cancer Wisconsin Diagnostic (Public Benchmark via Scikit-Learn)
- **Objective:** Design, compile, train, and evaluate a Deep Feedforward Neural Network to classify tumour biopsies as **Malignant** (0) or **Benign** (1).
- **Dataset Dimensions:** {raw_shape[0]} records, {len(feature_cols)} continuous input features.

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
- **Test Accuracy:** {metrics_dict['accuracy']:.4f} ({metrics_dict['accuracy']*100:.1f}%)
- **Test Precision:** {metrics_dict['precision']:.4f}
- **Test Recall:** {metrics_dict['recall']:.4f}
- **Test F1-Score:** {metrics_dict['f1']:.4f}
- **Test ROC-AUC:** {metrics_dict['roc_auc']:.4f}
- **Epochs Trained:** {metrics_dict['epochs_trained']} (Early Stopping converged smoothly)

## 5. Artifacts and Generated Deliverables
- **Models:** Saved to `models/` (`deep_classifier_model.keras` and `best_model_weights.weights.h5`)
- **Figures:** Saved to `figures/` (Fig 1: Training History Curves, Fig 2: Confusion Matrix, Fig 3: ROC Curve)
- **Metrics Summary:** Saved to `outputs/deep_learning_summary.json` and `data/processed/deep_learning_metrics.csv`
"""
    findings_path = os.path.join(OUT_DIR, "key_findings.md")
    with open(findings_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SAVED] {findings_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  WEEK 5 - DEEP LEARNING PIPELINE  |  TensorFlow & Keras")
    print("="*60)

    # 1. Acquire & Preprocess
    (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, raw_shape) = acquire_and_preprocess_data()

    # 2. Build Architecture
    model = build_model(input_dim=len(feature_cols))

    # 3. Compile Model
    model = compile_model(model)

    # 4. Train Model
    history, full_model_path, weights_path = train_model(model, X_train, y_train, X_val, y_val)

    # 5. Evaluate & Visualize
    metrics_dict = evaluate_and_visualize(model, history, X_test, y_test)

    # 6. Save Summaries & Artifacts
    save_summaries(metrics_dict, raw_shape, feature_cols)

    print("\n" + "="*60)
    print("  ALL DONE - Week 5 deep learning pipeline completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()
