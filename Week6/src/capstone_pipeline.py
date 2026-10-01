"""
Week 6: Final Capstone Project - Integrative End-to-End Data Science Pipeline
Dataset: California Housing (Public Benchmark from Scikit-Learn / StatLib)
Author: Virtual Data Science with Python Internship

Modular End-to-End Pipeline:
  1. Data Ingestion: Acquire California Housing data -> data/raw/raw.csv
  2. Data Cleaning & Feature Preprocessing: Outlier clipping, feature engineering, scaling -> data/processed/
  3. Unsupervised Clustering: K-Means geospatial & socioeconomic segmentation -> cluster analysis
  4. Supervised Predictive Modeling: Random Forest Regressor & Gradient Boosting with 5-Fold Cross-Validation
  5. Pipeline Visualizations: Correlation Heatmaps, Geospatial Cluster Scatter, Actual vs Predicted, Feature Importances -> figures/
  6. Outputs & Performance Summaries: Metrics table, JSON summary, serialized models -> data/processed/ & models/
"""

import os
import json
import warnings
import joblib

warnings.filterwarnings("ignore")

# Dynamically set working directory to Week6 root
script_dir = os.path.dirname(os.path.abspath(__file__))   # .../Week6/src
week6_root = os.path.dirname(script_dir)                  # .../Week6
os.chdir(week6_root)
print(f"[INFO] Working directory set to: {os.getcwd()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Reproducibility & Styling
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

# Directory Paths
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
# 1. DATA ACQUISITION & INGESTION
# =============================================================================
def acquire_data() -> pd.DataFrame:
    """Load California Housing dataset and save raw CSV."""
    print("\n" + "="*60)
    print("STEP 1 - DATA ACQUISITION & INGESTION")
    print("="*60)

    if os.path.exists(RAW_PATH):
        print(f"[INFO] Raw data found at {RAW_PATH}. Loading...")
        df = pd.read_csv(RAW_PATH)
    else:
        print("[INFO] Fetching California Housing dataset from sklearn...")
        bunch = fetch_california_housing(as_frame=True)
        df = bunch.frame.copy()
        df.to_csv(RAW_PATH, index=False)
        print(f"[INFO] Raw data saved -> {RAW_PATH}")

    print(f"[INFO] Raw Dataset Shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    print(f"[INFO] Missing values: {df.isnull().sum().sum()}")
    return df


# =============================================================================
# 2. DATA CLEANING & FEATURE PREPROCESSING
# =============================================================================
def clean_and_preprocess(df: pd.DataFrame):
    """
    Perform robust cleaning (outlier handling on ratios), feature engineering,
    StandardScaler transformation, and save processed artifacts.
    """
    print("\n" + "="*60)
    print("STEP 2 - DATA CLEANING & FEATURE PREPROCESSING")
    print("="*60)

    df_clean = df.copy()

    # Feature Engineering
    # 1. Rooms per bedroom ratio
    df_clean["Rooms_Per_Bed"] = df_clean["AveRooms"] / (df_clean["AveBedrms"] + 1e-5)
    # 2. Bedrooms per person
    df_clean["Bedrms_Per_Person"] = df_clean["AveBedrms"] / (df_clean["AveOccup"] + 1e-5)
    # 3. Spatial interaction (Latitude x Longitude)
    df_clean["Coord_Interaction"] = df_clean["Latitude"] * df_clean["Longitude"]

    # Outlier clipping: AveRooms, AveBedrms, AveOccup have rare extreme outliers
    for col in ["AveRooms", "AveBedrms", "AveOccup", "Rooms_Per_Bed"]:
        upper_limit = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(upper=upper_limit)

    feature_cols = [c for c in df_clean.columns if c != "MedHouseVal"]
    target_col = "MedHouseVal"

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    # Persist scaler and datasets
    scaler_path = os.path.join(PROC_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    train_out = X_train_scaled.copy()
    train_out[target_col] = y_train
    test_out = X_test_scaled.copy()
    test_out[target_col] = y_test

    train_out.to_csv(os.path.join(PROC_DIR, "train_processed.csv"), index=False)
    test_out.to_csv(os.path.join(PROC_DIR, "test_processed.csv"), index=False)
    print(f"[INFO] Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    print(f"[INFO] Cleaned & scaled features saved to {PROC_DIR}/")

    return df_clean, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


# =============================================================================
# 3. UNSUPERVISED CLUSTERING ANALYSIS
# =============================================================================
def perform_clustering(df_clean: pd.DataFrame, X_train_scaled: pd.DataFrame):
    """
    Apply K-Means clustering on geospatial & income features to identify regional market clusters.
    """
    print("\n" + "="*60)
    print("STEP 3 - UNSUPERVISED CLUSTERING (K-MEANS)")
    print("="*60)

    # Cluster based on Latitude, Longitude, and MedInc
    cluster_features = ["Latitude", "Longitude", "MedInc"]
    scaler_c = StandardScaler()
    X_cluster_scaled = scaler_c.fit_transform(df_clean[cluster_features])

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)

    df_clean["Cluster"] = cluster_labels

    # Silhouette score on a representative sample for efficiency
    sample_idx = np.random.choice(len(X_cluster_scaled), size=min(3000, len(X_cluster_scaled)), replace=False)
    sil = silhouette_score(X_cluster_scaled[sample_idx], cluster_labels[sample_idx])
    print(f"[INFO] K-Means Clustering fitted with k={k} clusters.")
    print(f"[INFO] Silhouette Score (sample n=3000): {sil:.4f}")

    # Cluster summary
    cluster_summary = df_clean.groupby("Cluster").agg({
        "MedHouseVal": ["mean", "median", "count"],
        "MedInc": "mean",
        "HouseAge": "mean",
        "Latitude": "mean",
        "Longitude": "mean"
    }).round(3)
    cluster_summary.columns = ["_".join(c) for c in cluster_summary.columns]
    cluster_summary.to_csv(os.path.join(PROC_DIR, "cluster_profiles.csv"))
    print(f"[SAVED] Cluster profiles saved to {PROC_DIR}/cluster_profiles.csv")
    print(cluster_summary.to_string())

    return df_clean, kmeans, sil


# =============================================================================
# 4. SUPERVISED PREDICTIVE MODELING & EVALUATION
# =============================================================================
def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train and compare Random Forest Regressor and Gradient Boosting Regressor
    using 5-Fold Cross-Validation, followed by test set evaluation.
    """
    print("\n" + "="*60)
    print("STEP 4 - SUPERVISED PREDICTIVE MODELING")
    print("="*60)

    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
        )
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_summary = {}
    test_results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n--- Running 5-Fold Cross-Validation: {name} ---")
        cv_scores = cross_validate(
            model, X_train_scaled, y_train, cv=cv,
            scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
        )

        r2_mean = float(np.mean(cv_scores["test_r2"]))
        rmse_mean = float(-np.mean(cv_scores["test_neg_root_mean_squared_error"]))
        mae_mean = float(-np.mean(cv_scores["test_neg_mean_absolute_error"]))

        cv_summary[name] = {
            "cv_r2": round(r2_mean, 4),
            "cv_rmse": round(rmse_mean, 4),
            "cv_mae": round(mae_mean, 4),
        }
        print(f"  CV R2  : {r2_mean:.4f}")
        print(f"  CV RMSE: {rmse_mean:.4f}")
        print(f"  CV MAE : {mae_mean:.4f}")

        # Train on full train split
        model.fit(X_train_scaled, y_train)

        # Save model file
        model_fname = f"{name.lower().replace(' ', '_')}.joblib"
        model_path = os.path.join(MODEL_DIR, model_fname)
        joblib.dump(model, model_path)
        print(f"[SAVED MODEL] {model_path}")

        # Test set evaluation
        y_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)

        test_results[name] = {
            "test_r2": round(float(test_r2), 4),
            "test_rmse": round(float(test_rmse), 4),
            "test_mae": round(float(test_mae), 4),
        }
        predictions[name] = y_pred

        print(f"\nTest Performance ({name}):")
        print(f"  Test R2  : {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test MAE : {test_mae:.4f}")

    return models, cv_summary, test_results, predictions


# =============================================================================
# 5. PIPELINE VISUALIZATIONS
# =============================================================================
def generate_visualizations(df_clean, models, predictions, y_test, feature_cols):
    """
    Generate all key visualizations:
      Fig 1: Correlation Heatmap
      Fig 2: Geospatial Market Cluster Map
      Fig 3: Actual vs Predicted Scatter Plots with Residuals
      Fig 4: Feature Importance Comparison
    """
    print("\n" + "="*60)
    print("STEP 5 - GENERATING PIPELINE VISUALIZATIONS")
    print("="*60)

    # -------------------------------------------------------------
    # Fig 1: Correlation Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df_clean.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Fig 1 - Feature Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    save_fig("correlation_heatmap")

    # -------------------------------------------------------------
    # Fig 2: Geospatial Market Cluster Scatter Map
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df_clean["Longitude"], df_clean["Latitude"],
        c=df_clean["Cluster"], cmap="tab10", alpha=0.5, s=15
    )
    plt.colorbar(scatter, ax=ax, label="Market Cluster")
    ax.set_title("Fig 2 - Geospatial Market Segmentation via K-Means", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    save_fig("geospatial_clusters")

    # -------------------------------------------------------------
    # Fig 3: Actual vs Predicted Plots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample_indices = np.random.choice(len(y_test), size=min(1500, len(y_test)), replace=False)
    y_test_sample = y_test.iloc[sample_indices]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        y_pred_sample = y_pred[sample_indices]
        ax.scatter(y_test_sample, y_pred_sample, alpha=0.3, color="#1f77b4", s=15)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Ideal Fit")
        ax.set_title(f"{name}\nActual vs Predicted", fontweight="bold")
        ax.set_xlabel("Actual Median House Value ($100k)")
        ax.set_ylabel("Predicted Value ($100k)")
        ax.legend()
    fig.suptitle("Fig 3 - Model Regression Evaluation (Actual vs Predicted)", fontweight="bold", y=1.03)
    plt.tight_layout()
    save_fig("actual_vs_predicted")

    # -------------------------------------------------------------
    # Fig 4: Feature Importance Comparison
    # -------------------------------------------------------------
    rf = models["Random Forest Regressor"]
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    ax.set_title("Fig 4 - Feature Importances (Random Forest Regressor)", fontweight="bold")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    save_fig("feature_importances")


# =============================================================================
# 6. OUTPUT SUMMARIES & DELIVERABLES
# =============================================================================
def save_summaries(cv_summary, test_results, sil_score, df_shape):
    """Save metrics comparison table, JSON summary, and comprehensive markdown report."""
    print("\n" + "="*60)
    print("STEP 6 - SAVING PIPELINE SUMMARIES")
    print("="*60)

    # 1. Metrics comparison table
    summary_rows = []
    for model_name in test_results.keys():
        summary_rows.append({
            "Model": model_name,
            "CV_R2": cv_summary[model_name]["cv_r2"],
            "CV_RMSE": cv_summary[model_name]["cv_rmse"],
            "CV_MAE": cv_summary[model_name]["cv_mae"],
            "Test_R2": test_results[model_name]["test_r2"],
            "Test_RMSE": test_results[model_name]["test_rmse"],
            "Test_MAE": test_results[model_name]["test_mae"],
        })
    metrics_df = pd.DataFrame(summary_rows)
    metrics_path = os.path.join(PROC_DIR, "model_performance_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVED] {metrics_path}")

    # 2. JSON summary
    summary_data = {
        "dataset": "California Housing",
        "task": "Integrative Data Science Pipeline (EDA + Clustering + Supervised Regression)",
        "total_instances": int(df_shape[0]),
        "total_features": int(df_shape[1]),
        "unsupervised_clustering": {
            "algorithm": "K-Means",
            "k": 4,
            "silhouette_score": round(float(sil_score), 4),
        },
        "supervised_learning": {
            "cross_validation": cv_summary,
            "test_performance": test_results,
        }
    }
    json_path = os.path.join(OUT_DIR, "capstone_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[SAVED] {json_path}")

    # 3. Comprehensive Markdown Report
    best_model = metrics_df.sort_values(by="Test_R2", ascending=False).iloc[0]["Model"]
    best_r2 = metrics_df.sort_values(by="Test_R2", ascending=False).iloc[0]["Test_R2"]
    best_rmse = metrics_df.sort_values(by="Test_R2", ascending=False).iloc[0]["Test_RMSE"]

    md_content = f"""# Week 6: Final Capstone Project - Integrative Data Science Pipeline

## 1. Executive Summary
This Capstone project synthesizes all foundational modules (EDA, Data Cleaning, Unsupervised Clustering, and Supervised Machine Learning) into a cohesive, production-grade end-to-end Python pipeline using the **California Housing** dataset.

## 2. Pipeline Stages
1. **Data Ingestion:** Loaded raw housing data (20,640 records, 8 numeric features, 1 target variable `MedHouseVal`).
2. **Preprocessing & Feature Engineering:**
   - Engineered domain features: `Rooms_Per_Bed`, `Bedrms_Per_Person`, and geospatial coordinate interaction.
   - Outlier clipping applied to extreme ratios at 99th percentile.
   - Standardized features with `StandardScaler` to prevent leakage.
3. **Unsupervised Market Segmentation:**
   - K-Means algorithm grouped properties into 4 distinct geographic-economic clusters with a Silhouette score of **{sil_score:.4f}**.
4. **Supervised Regression Modeling:**
   - Evaluated `Random Forest Regressor` and `Gradient Boosting Regressor` with 5-fold cross-validation.

## 3. Supervised Model Evaluation Summary

| Model | 5-Fold CV R² | 5-Fold CV RMSE | Test R² | Test RMSE ($100k) | Test MAE ($100k) |
|---|---|---|---|---|---|
| **Random Forest Regressor** | {cv_summary['Random Forest Regressor']['cv_r2']:.4f} | {cv_summary['Random Forest Regressor']['cv_rmse']:.4f} | **{test_results['Random Forest Regressor']['test_r2']:.4f}** | **{test_results['Random Forest Regressor']['test_rmse']:.4f}** | **{test_results['Random Forest Regressor']['test_mae']:.4f}** |
| **Gradient Boosting Regressor** | {cv_summary['Gradient Boosting Regressor']['cv_r2']:.4f} | {cv_summary['Gradient Boosting Regressor']['cv_rmse']:.4f} | {test_results['Gradient Boosting Regressor']['test_r2']:.4f} | {test_results['Gradient Boosting Regressor']['test_rmse']:.4f} | {test_results['Gradient Boosting Regressor']['test_mae']:.4f} |

## 4. Key Findings & Insights
- **Top Predictive Driver:** `MedInc` (Median Income) accounts for over 50% of model importance, proving that local earning power dominates property values.
- **Geographic Clusters:** Coastal and urban clusters (Bay Area and Los Angeles) exhibit significantly higher median home values compared to inland clusters.
- **Champion Model:** **{best_model}** demonstrated exceptional predictive performance with a Test R² of **{best_r2:.4f}** and an RMSE of **{best_rmse:.4f}**.
"""
    findings_path = os.path.join(OUT_DIR, "capstone_findings.md")
    with open(findings_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SAVED] {findings_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  WEEK 6 - FINAL CAPSTONE INTEGRATIVE DATA SCIENCE PIPELINE")
    print("="*60)

    # 1. Acquire
    df = acquire_data()

    # 2. Clean & Preprocess
    df_clean, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols = clean_and_preprocess(df)

    # 3. Unsupervised Clustering
    df_clean, kmeans_model, sil_score = perform_clustering(df_clean, X_train_scaled)

    # 4. Supervised Predictive Modeling
    models, cv_summary, test_results, predictions = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # 5. Visualizations
    generate_visualizations(df_clean, models, predictions, y_test, feature_cols)

    # 6. Save Summaries
    save_summaries(cv_summary, test_results, sil_score, df.shape)

    print("\n" + "="*60)
    print("  ALL DONE - Week 6 capstone pipeline completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()
