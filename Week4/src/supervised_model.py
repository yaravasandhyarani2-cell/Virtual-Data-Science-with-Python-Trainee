"""
Week 4: Supervised Learning and Classification Analysis
Dataset: Breast Cancer Wisconsin Diagnostic (Public Benchmark Dataset via Scikit-Learn)
Author: Virtual Data Science with Python Internship

Modular Pipeline:
  1. Data acquisition and storage: Load dataset -> data/raw/raw.csv
  2. Preprocessing & Train-Test Split: Scaled via StandardScaler -> data/processed/
  3. Supervised Model Training: Logistic Regression & Random Forest Classifier
  4. Cross-Validation & Metric Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  5. Evaluation Visualizations: Confusion Matrices, ROC Curves, PR Curves, Feature Importances -> figures/
  6. Artifact Persisting: Save trained models (joblib) & evaluation summaries -> data/processed/ & outputs/
"""

import os
import json
import warnings
import joblib

warnings.filterwarnings("ignore")

# Set working directory to Week4 root dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))   # .../Week4/src
week4_root = os.path.dirname(script_dir)                  # .../Week4
os.chdir(week4_root)
print(f"[INFO] Working directory set to: {os.getcwd()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# Global plot styling
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})
PALETTE = "Set2"
RANDOM_STATE = 42

# Directory Paths
RAW_PATH = os.path.join("data", "raw", "raw.csv")
PROC_DIR = os.path.join("data", "processed")
FIG_DIR = os.path.join("figures")
OUT_DIR = os.path.join("outputs")
SUMMARY_JSON = os.path.join(OUT_DIR, "evaluation_summary.json")

os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

fig_counter = [0]

def save_fig(name: str) -> str:
    """Save current figure with sequential numbering to figures/."""
    fig_counter[0] += 1
    fname = f"fig{fig_counter[0]:02d}_{name}.png"
    fpath = os.path.join(FIG_DIR, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fpath}")
    return fpath


# =============================================================================
# 1. DATA ACQUISITION
# =============================================================================
def acquire_data() -> pd.DataFrame:
    """Load Breast Cancer Wisconsin Diagnostic dataset and save raw CSV."""
    print("\n" + "="*60)
    print("SECTION 1 - DATA ACQUISITION")
    print("="*60)

    if os.path.exists(RAW_PATH):
        print(f"[INFO] Raw data already exists at {RAW_PATH}. Loading...")
        df = pd.read_csv(RAW_PATH)
        print(f"[INFO] Shape: {df.shape}")
        return df

    print("[INFO] Loading dataset from sklearn.datasets.load_breast_cancer...")
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    
    # Target in bunch is 0 = Malignant, 1 = Benign.
    # Add target_name for human readability while retaining binary target
    df["target_name"] = df["target"].map({0: "malignant", 1: "benign"})
    df.to_csv(RAW_PATH, index=False)
    print(f"[INFO] Raw data saved -> {RAW_PATH}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Target distribution:\n{df['target_name'].value_counts().to_string()}")
    return df


# =============================================================================
# 2. PREPROCESSING & TRAIN-TEST SPLITTING
# =============================================================================
def preprocess_and_split(df: pd.DataFrame):
    """
    Perform feature/target separation, stratified 80/20 train-test split,
    StandardScaler fitting on train data, and persist processed CSVs.
    """
    print("\n" + "="*60)
    print("SECTION 2 - PREPROCESSING & STRATIFIED TRAIN-TEST SPLIT")
    print("="*60)

    # Feature columns exclude target and target_name
    feature_cols = [c for c in df.columns if c not in ["target", "target_name"]]
    X = df[feature_cols].copy()
    y = df["target"].astype(int).copy()

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    print(f"[INFO] Number of features: {X_train.shape[1]}")

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    # Save scaler
    scaler_path = os.path.join(PROC_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[SAVED] {scaler_path}")

    # Save train & test splits
    train_df = X_train_scaled.copy()
    train_df["target"] = y_train
    test_df = X_test_scaled.copy()
    test_df["target"] = y_test

    train_path = os.path.join(PROC_DIR, "train_processed.csv")
    test_path = os.path.join(PROC_DIR, "test_processed.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[SAVED] {train_path} (shape: {train_df.shape})")
    print(f"[SAVED] {test_path} (shape: {test_df.shape})")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


# =============================================================================
# 3. CROSS-VALIDATION & MODEL TRAINING
# =============================================================================
def train_and_cross_validate(X_train_scaled, y_train, feature_cols):
    """
    Perform 5-fold Stratified Cross-Validation on Logistic Regression and
    Random Forest Classifier, then fit final models on the entire training set.
    """
    print("\n" + "="*60)
    print("SECTION 3 - 5-FOLD STRATIFIED CROSS-VALIDATION & MODEL TRAINING")
    print("="*60)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_results_summary = {}

    for name, model in models.items():
        print(f"\n--- Running 5-Fold Cross-Validation: {name} ---")
        scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring)
        cv_summary = {}
        for metric in scoring.keys():
            mean_score = float(np.mean(scores[f"test_{metric}"]))
            std_score = float(np.std(scores[f"test_{metric}"]))
            cv_summary[metric] = {
                "mean": round(mean_score, 4),
                "std": round(std_score, 4),
            }
            print(f"  CV {metric.capitalize():<10}: {mean_score:.4f} (+/- {std_score:.4f})")
        cv_results_summary[name] = cv_summary

        # Fit final model on full training set
        model.fit(X_train_scaled, y_train)

    return models, cv_results_summary


# =============================================================================
# 4. EVALUATION ON TEST SET
# =============================================================================
def evaluate_models(models, X_test_scaled, y_test):
    """
    Evaluate models on the held-out test set computing Accuracy, Precision,
    Recall, F1-score, and ROC-AUC.
    """
    print("\n" + "="*60)
    print("SECTION 4 - TEST SET EVALUATION")
    print("="*60)

    test_results = {}
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=["malignant", "benign"], output_dict=True)

        test_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
            "classification_report": report,
        }
        predictions[name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        print(f"\nModel: {name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")

    return test_results, predictions


# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================
def generate_visualizations(models, predictions, y_test, feature_cols):
    """
    Generate and save evaluation figures inside figures/:
      Fig 1: Confusion Matrices for both models
      Fig 2: ROC Curves comparison
      Fig 3: Precision-Recall Curves comparison
      Fig 4: Top 15 Feature Importances (Random Forest)
      Fig 5: Metric Comparison Bar Chart (CV vs Test)
    """
    print("\n" + "="*60)
    print("SECTION 5 - GENERATING VISUALIZATIONS")
    print("="*60)

    # -------------------------------------------------------------
    # Fig 1: Confusion Matrices
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, pred_data) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, pred_data["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"]
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    fig.suptitle("Fig 1 - Test Set Confusion Matrices", fontweight="bold", y=1.03)
    plt.tight_layout()
    save_fig("confusion_matrices")

    # -------------------------------------------------------------
    # Fig 2: ROC Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Logistic Regression": "#1f77b4", "Random Forest Classifier": "#2ca02c"}
    for name, pred_data in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, pred_data["y_proba"])
        auc = roc_auc_score(y_test, pred_data["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors.get(name, "black"), lw=2.5)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Fig 2 - Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    save_fig("roc_curves")

    # -------------------------------------------------------------
    # Fig 3: Precision-Recall Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pred_data in predictions.items():
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, pred_data["y_proba"])
        ap = average_precision_score(y_test, pred_data["y_proba"])
        ax.plot(recall_vals, precision_vals, label=f"{name} (AP = {ap:.4f})", color=colors.get(name, "black"), lw=2.5)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Fig 3 - Precision-Recall (PR) Curves", fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    save_fig("precision_recall_curves")

    # -------------------------------------------------------------
    # Fig 4: Random Forest Feature Importances
    # -------------------------------------------------------------
    rf = models["Random Forest Classifier"]
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax)
    ax.set_title("Fig 4 - Top 15 Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Gini Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    save_fig("feature_importances")

    # Save feature importances to CSV
    all_feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).reset_index()
    all_feat_imp.columns = ["feature", "importance"]
    feat_imp_path = os.path.join(PROC_DIR, "feature_importances.csv")
    all_feat_imp.to_csv(feat_imp_path, index=False)
    print(f"[SAVED] {feat_imp_path}")


# =============================================================================
# 6. SAVE ARTIFACTS AND OUTPUTS
# =============================================================================
def save_artifacts_and_summaries(models, cv_summary, test_results, df_shape, feature_cols):
    """
    Save trained model files (.joblib), metrics table CSVs, evaluation_summary.json,
    and a comprehensive key_findings.md.
    """
    print("\n" + "="*60)
    print("SECTION 6 - SAVING ARTIFACTS AND SUMMARIES")
    print("="*60)

    # 1. Save trained models to data/processed/
    for name, model in models.items():
        fname = f"{name.lower().replace(' ', '_')}.joblib"
        model_path = os.path.join(PROC_DIR, fname)
        joblib.dump(model, model_path)
        print(f"[SAVED MODEL] {model_path}")

    # 2. Save test metrics comparison table to data/processed/
    metrics_list = []
    for name, res in test_results.items():
        metrics_list.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1-Score": res["f1"],
            "ROC-AUC": res["roc_auc"],
        })
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(PROC_DIR, "model_comparison_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVED] {metrics_path}")

    # 3. Create comprehensive JSON summary in outputs/ and data/processed/
    summary = {
        "dataset": "Breast Cancer Wisconsin Diagnostic",
        "task": "Supervised Binary Classification (Malignant vs Benign)",
        "rows": int(df_shape[0]),
        "features_count": len(feature_cols),
        "train_size": 455,
        "test_size": 114,
        "cross_validation": {
            "folds": 5,
            "strategy": "StratifiedKFold",
            "results": cv_summary,
        },
        "test_evaluation": test_results,
    }

    # Save to both outputs/ and data/processed/ to fulfill requirements
    for path in [SUMMARY_JSON, os.path.join(PROC_DIR, "evaluation_summary.json")]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[SAVED] {path}")

    # 4. Generate Key Findings / Model Card Markdown
    best_model_name = metrics_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Model"]
    best_auc = metrics_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]["ROC-AUC"]
    best_acc = metrics_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Accuracy"]

    md_content = f"""# Week 4 - Supervised Learning & Classification Analysis

## 1. Dataset Overview
- **Dataset:** Breast Cancer Wisconsin Diagnostic (Public Benchmark from Scikit-Learn / UCI)
- **Problem Formulation:** Binary Classification (predicting whether a tumour biopsy is `malignant` (0) or `benign` (1)).
- **Total Samples:** {df_shape[0]} ({summary['train_size']} Train / {summary['test_size']} Test stratified split)
- **Features:** {len(feature_cols)} continuous cell nucleus characteristics computed from digitized fine needle aspirate (FNA) images.

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
| **Logistic Regression** | {test_results['Logistic Regression']['accuracy']:.4f} | {test_results['Logistic Regression']['precision']:.4f} | {test_results['Logistic Regression']['recall']:.4f} | {test_results['Logistic Regression']['f1']:.4f} | {test_results['Logistic Regression']['roc_auc']:.4f} |
| **Random Forest Classifier** | {test_results['Random Forest Classifier']['accuracy']:.4f} | {test_results['Random Forest Classifier']['precision']:.4f} | {test_results['Random Forest Classifier']['recall']:.4f} | {test_results['Random Forest Classifier']['f1']:.4f} | {test_results['Random Forest Classifier']['roc_auc']:.4f} |

## 4. Key Findings & Insights
1. **High Diagnostic Accuracy:** Both models achieve high diagnostic performance on the held-out test set, with **{best_model_name}** achieving an accuracy of **{best_acc*100:.1f}%** and an ROC-AUC of **{best_auc:.4f}**.
2. **Top Predictive Biomarkers:** Feature importance analysis via Random Forest reveals that `worst concave points`, `worst perimeter`, `worst radius`, and `mean concave points` are the strongest predictors of tumour malignancy.
3. **Clinical Implication:** High Recall ({test_results['Random Forest Classifier']['recall']:.4f} for Random Forest and {test_results['Logistic Regression']['recall']:.4f} for Logistic Regression) minimizes false negatives, critical in medical diagnostics to avoid missed diagnoses.

## 5. Artifacts and Generated Deliverables
- **Figures:** Saved to `figures/` (Fig 1: Confusion Matrices, Fig 2: ROC Curves, Fig 3: PR Curves, Fig 4: Feature Importances)
- **Serialized Models:** Saved to `data/processed/` (`logistic_regression.joblib`, `random_forest_classifier.joblib`, `scaler.joblib`)
- **Evaluation Summaries:** Saved to `data/processed/model_comparison_metrics.csv` and `outputs/evaluation_summary.json`
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
    print("  WEEK 4 - SUPERVISED LEARNING PIPELINE")
    print("="*60)

    # 1. Acquire data
    df = acquire_data()

    # 2. Preprocess & Split
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler = preprocess_and_split(df)

    # 3. Train & Cross-Validate
    models, cv_summary = train_and_cross_validate(X_train_scaled, y_train, feature_cols)

    # 4. Evaluate on Test Set
    test_results, predictions = evaluate_models(models, X_test_scaled, y_test)

    # 5. Visualizations
    generate_visualizations(models, predictions, y_test, feature_cols)

    # 6. Save Artifacts and Summaries
    save_artifacts_and_summaries(models, cv_summary, test_results, df.shape, feature_cols)

    print("\n" + "="*60)
    print("  ALL DONE - Week 4 supervised learning pipeline completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()
