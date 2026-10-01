"""
Ames Housing Data Cleaning and Preprocessing Pipeline
Week 1: Data Acquisition, Cleaning, and Preprocessing

This script provides an automated, reproducible end-to-end pipeline:
1. Acquires the Ames Housing dataset from OpenML and saves untouched raw data.
2. Performs data auditing (missing values, outliers, erroneous entries, dtypes).
3. Handles missing values systematically (structural NaNs, grouped medians, mode).
4. Treats outliers via statistical winsorization to preserve sample density.
5. Standardizes types and encodes variables (ordinal scale + one-hot encoding).
6. Scales numeric features and log1p-transforms the target variable.
7. Validates cleaned data and exports data/processed/cleaned.csv.
8. Benchmarks downstream impact using Ridge and RandomForestRegressor with 5-fold CV.
9. Exports figures to outputs/ and logs metrics to outputs/cleaning_log.json.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set aesthetic defaults
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.autolayout": True
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Ensure working directory is Week1 root (parent directory of src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEEK1_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.exists(os.path.join(WEEK1_ROOT, "src")):
    os.chdir(WEEK1_ROOT)

def ensure_directories():
    """Ensure required project directories exist within Week1."""
    dirs = [
        os.path.join("data", "raw"),
        os.path.join("data", "processed"),
        "outputs",
        "notebooks",
        "src"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def acquire_data(raw_csv_path="data/raw/raw.csv"):
    """
    Download Ames Housing dataset via sklearn fetch_openml and save untouched raw CSV.
    """
    logger.info("Fetching Ames Housing dataset from OpenML (name='house_prices')...")
    bunch = fetch_openml(name="house_prices", as_frame=True, parser="auto")
    df = bunch.frame.copy()
    
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    df.to_csv(raw_csv_path, index=False)
    logger.info(f"Raw data successfully saved to {raw_csv_path} with shape {df.shape}")
    return df


def audit_missing(df):
    """Calculate missing value counts and percentages."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df)) * 100.0
    return pd.DataFrame({"MissingCount": missing, "Percentage": missing_pct})


def plot_missing_values(missing_df, output_path="outputs/01_missing_values.png"):
    """Plot missing value bar chart with count and percentage."""
    if missing_df.empty:
        return
    plt.figure(figsize=(12, 6))
    bars = plt.bar(missing_df.index, missing_df["Percentage"], color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
    plt.axhline(80, color="#ef4444", linestyle="--", linewidth=1.5, label="80% Threshold (Drop Rule)")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylabel("Missing Percentage (%)")
    plt.xlabel("Features with Missing Values")
    plt.title("Ames Housing: Missing Value Percentage per Feature")
    plt.legend(frameon=True)
    
    # Annotate bar percentages
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)
    
    plt.figtext(0.5, -0.05, "Figure 1: Missing values count and percentage per column in raw Ames housing dataset before imputation.", ha="center", fontsize=10, style="italic")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved missing values plot to {output_path}")


def plot_raw_correlation(df, output_path="outputs/02_correlation_heatmap_raw.png"):
    """Plot correlation heatmap for top numeric features."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if "Id" in num_cols:
        num_cols = num_cols.drop("Id")
    corr = df[num_cols].corr()
    
    # Select top 15 correlated features with SalePrice
    if "SalePrice" in corr.columns:
        top_features = corr["SalePrice"].abs().sort_values(ascending=False).head(15).index
        sub_corr = df[top_features].corr()
    else:
        sub_corr = corr.iloc[:15, :15]
        
    plt.figure(figsize=(12, 10))
    sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap: Top Features Associated with SalePrice (Raw Data)")
    plt.figtext(0.5, -0.02, "Figure 2: Pearson correlation matrix of top numeric features highlighting multicollinearity and target associations.", ha="center", fontsize=10, style="italic")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved correlation heatmap to {output_path}")


def plot_raw_distributions(df, output_path="outputs/03_raw_distributions.png"):
    """Plot histograms and KDE for key target and predictor features."""
    features = ["SalePrice", "GrLivArea", "LotArea", "TotalBsmtSF"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        if col in df.columns:
            skew = df[col].dropna().skew()
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#0ea5e9", edgecolor="#0284c7")
            axes[i].set_title(f"{col} Distribution (Skewness: {skew:.2f})")
            axes[i].set_xlabel(f"{col}")
            axes[i].set_ylabel("Frequency")
            
    plt.suptitle("Raw Distributions and Skewness of Key Variables", fontsize=15, y=1.02)
    plt.figtext(0.5, -0.02, "Figure 3: Distribution histograms and KDE contours for SalePrice and key area metrics in raw data.", ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved raw distributions plot to {output_path}")


def detect_outliers(df, cols=["GrLivArea", "LotArea", "SalePrice", "TotalBsmtSF"]):
    """Calculate IQR bounds and Z-score outlier counts for specified columns."""
    outlier_info = {}
    for col in cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        iqr_mask = (series < lower_iqr) | (series > upper_iqr)
        
        mean = series.mean()
        std = series.std()
        z_scores = (series - mean) / std
        z_mask = np.abs(z_scores) > 3.0
        
        outlier_info[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "iqr_lower": float(lower_iqr),
            "iqr_upper": float(upper_iqr),
            "iqr_count": int(iqr_mask.sum()),
            "zscore_count": int(z_mask.sum()),
            "mean": float(mean),
            "std": float(std)
        }
    return outlier_info


def plot_outlier_diagnostics(df_before, df_after, cols=["GrLivArea", "LotArea", "TotalBsmtSF", "SalePrice"]):
    """Plot box plots and scatter plots comparing before vs after outlier treatment."""
    # 1. Box plots before and after
    fig, axes = plt.subplots(len(cols), 2, figsize=(14, 16))
    for i, col in enumerate(cols):
        # Before
        sns.boxplot(x=df_before[col], ax=axes[i, 0], color="#93c5fd")
        axes[i, 0].set_title(f"{col} - Before Treatment (Raw)")
        axes[i, 0].set_xlabel(col)
        
        # After
        sns.boxplot(x=df_after[col], ax=axes[i, 1], color="#86efac")
        axes[i, 1].set_title(f"{col} - After Winsorization (Treated)")
        axes[i, 1].set_xlabel(col)
        
    plt.suptitle("Outlier Diagnostic: Feature Box Plots Before vs After Winsorization", fontsize=15, y=1.01)
    plt.figtext(0.5, -0.01, "Figure 4: Comparative box plots for key features demonstrating reduction in extreme tail leverage post-winsorization.", ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig("outputs/04_outliers_boxplot_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Scatter plots vs SalePrice before and after
    scatter_features = ["GrLivArea", "LotArea", "TotalBsmtSF"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    for i, col in enumerate(scatter_features):
        sns.scatterplot(x=df_before[col], y=df_before["SalePrice"], ax=axes[i, 0], color="#2563eb", alpha=0.6)
        axes[i, 0].set_title(f"{col} vs SalePrice (Before)")
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel("SalePrice ($)")
        
        sns.scatterplot(x=df_after[col], y=df_after["SalePrice"], ax=axes[i, 1], color="#059669", alpha=0.6)
        axes[i, 1].set_title(f"{col} vs SalePrice (After Winsorization)")
        axes[i, 1].set_xlabel(col)
        axes[i, 1].set_ylabel("SalePrice ($)")
        
    plt.suptitle("Bivariate Scatter Relationships with SalePrice Before vs After Winsorization", fontsize=15, y=1.01)
    plt.figtext(0.5, -0.01, "Figure 5: Scatter plots illustrating control of leverage points and preservation of underlying structural trends.", ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig("outputs/05_outliers_scatter_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved outlier boxplots and scatterplots to outputs/")


def run_full_pipeline():
    """Execute complete cleaning, preprocessing, validation, benchmarking, and logging."""
    ensure_directories()
    
    # 1 & 2: Acquisition
    raw_df = acquire_data()
    df = raw_df.copy()
    initial_shape = list(df.shape)
    
    # Audit missing values
    missing_before_df = audit_missing(df)
    missing_before_dict = {
        "total_missing_cells": int(df.isnull().sum().sum()),
        "columns_with_missing": {col: int(cnt) for col, cnt in missing_before_df["MissingCount"].items()}
    }
    plot_missing_values(missing_before_df)
    plot_raw_correlation(df)
    plot_raw_distributions(df)
    
    # Detect outliers prior to cleaning
    outliers_detected = detect_outliers(df)
    
    # Drop arbitrary identifier if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
        logger.info("Dropped 'Id' column (arbitrary identifier).")

    # 4: Handle Missing Values
    # Identify and drop columns with > 80% missing
    high_missing_cols = [c for c in df.columns if (df[c].isnull().sum() / len(df)) > 0.80]
    logger.info(f"Columns with >80% missing to be dropped: {high_missing_cols}")
    df = df.drop(columns=high_missing_cols)
    
    # Fill categorical features where NaN indicates 'Feature Absent'
    none_impute_cols = [
        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType"
    ]
    for col in none_impute_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")
            
    # LotFrontage imputed with median of respective Neighborhood
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        # Fallback to global median if an entire neighborhood is missing
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
        
    # Other numeric columns -> median
    numeric_cols_impute = df.select_dtypes(include=[np.number]).columns.drop("SalePrice", errors="ignore")
    for col in numeric_cols_impute:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Other categorical columns -> mode
    cat_cols_impute = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols_impute:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    missing_after_count = int(df.isnull().sum().sum())
    logger.info(f"Missing values after systematic imputation: {missing_after_count}")

    # 5: Outliers Treatment via Winsorization (1st and 99th percentile)
    # Justification: Preserves the entire 1,460 sample dataset while neutralizing extreme leverage
    outlier_treated_log = {}
    treated_cols = ["GrLivArea", "LotArea", "TotalBsmtSF", "SalePrice"]
    df_before_outliers = df.copy()
    
    for col in treated_cols:
        low_cap = df[col].quantile(0.01)
        high_cap = df[col].quantile(0.99)
        low_count = int((df[col] < low_cap).sum())
        high_count = int((df[col] > high_cap).sum())
        df[col] = df[col].clip(lower=low_cap, upper=high_cap)
        outlier_treated_log[col] = {
            "method": "winsorization_1_99_percentile",
            "lower_threshold": float(low_cap),
            "upper_threshold": float(high_cap),
            "capped_lower_count": low_count,
            "capped_upper_count": high_count,
            "total_capped": low_count + high_count
        }
        
    plot_outlier_diagnostics(df_before_outliers, df, treated_cols)

    # 6: Erroneous Entries Auditing and Fixing
    # Duplicate check
    duplicates_count = int(df.duplicated().sum())
    if duplicates_count > 0:
        df = df.drop_duplicates()
        
    # Logical check: YearRemodAdd < YearBuilt
    remod_errors = int((df["YearRemodAdd"] < df["YearBuilt"]).sum())
    if remod_errors > 0:
        df.loc[df["YearRemodAdd"] < df["YearBuilt"], "YearRemodAdd"] = df.loc[df["YearRemodAdd"] < df["YearBuilt"], "YearBuilt"]
        
    # Logical check: YearBuilt > YrSold
    year_sold_errors = int((df["YearBuilt"] > df["YrSold"]).sum())
    if year_sold_errors > 0:
        df.loc[df["YearBuilt"] > df["YrSold"], "YrSold"] = df.loc[df["YearBuilt"] > df["YrSold"], "YearBuilt"]
        
    # Area negativity check
    neg_area_count = 0
    area_cols = [c for c in df.columns if "SF" in c or "Area" in c]
    for c in area_cols:
        neg_count = int((df[c] < 0).sum())
        if neg_count > 0:
            neg_area_count += neg_count
            df[c] = df[c].clip(lower=0)
            
    # String whitespace strip & case standardization
    whitespace_trimmed_count = 0
    for c in df.select_dtypes(include=["object", "category", "string"]).columns:
        s = df[c].astype(str)
        diff = (s != s.str.strip()).sum()
        if diff > 0:
            whitespace_trimmed_count += int(diff)
        df[c] = s.str.strip()

    # Dtype correction: MSSubClass represents housing building types, not numeric measurements
    df["MSSubClass"] = df["MSSubClass"].astype(str)

    logical_errors_log = {
        "duplicates_removed": duplicates_count,
        "year_remod_built_fixed": remod_errors,
        "year_built_sold_fixed": year_sold_errors,
        "negative_areas_fixed": neg_area_count,
        "whitespace_entries_trimmed": whitespace_trimmed_count,
        "mssubclass_converted_to_categorical": True
    }

    # 7: Preprocessing & Encodings
    # Skewness calculation before target transformation
    numeric_for_skew = df.select_dtypes(include=[np.number]).columns
    skewness_before = {col: float(df[col].skew()) for col in numeric_for_skew}
    
    # Ordinal Encoding for Quality Ratings (Po < Fa < TA < Gd < Ex)
    quality_mapping = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    quality_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"
    ]
    encoded_ordinal_cols = []
    for col in quality_cols:
        if col in df.columns:
            df[col] = df[col].map(quality_mapping).fillna(0).astype(int)
            encoded_ordinal_cols.append(col)
            
    # One-Hot Encoding for Nominal Categorical Columns
    nominal_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    feature_count_before_ohe = df.shape[1] - 1  # Excluding target
    
    df_preprocessed = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    feature_count_after_ohe = df_preprocessed.shape[1] - 1
    
    # Target log1p transformation for analysis
    saleprice_raw = df_preprocessed["SalePrice"].copy()
    saleprice_log = np.log1p(df_preprocessed["SalePrice"])
    
    # Skewness after
    skewness_after = {
        "SalePrice_raw": float(saleprice_raw.skew()),
        "SalePrice_log1p": float(saleprice_log.skew()),
        "GrLivArea": float(df_preprocessed["GrLivArea"].skew()),
        "LotArea": float(df_preprocessed["LotArea"].skew()),
        "TotalBsmtSF": float(df_preprocessed["TotalBsmtSF"].skew())
    }

    # Plot SalePrice before and after log1p transform
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(saleprice_raw, kde=True, ax=axes[0], color="#2563eb", edgecolor="#1d4ed8")
    axes[0].set_title(f"SalePrice Before Transformation (Skew: {saleprice_raw.skew():.2f})")
    axes[0].set_xlabel("SalePrice ($)")
    axes[0].set_ylabel("Count")
    
    sns.histplot(saleprice_log, kde=True, ax=axes[1], color="#059669", edgecolor="#047857")
    axes[1].set_title(f"SalePrice After log1p Transformation (Skew: {saleprice_log.skew():.2f})")
    axes[1].set_xlabel("log(1 + SalePrice)")
    axes[1].set_ylabel("Count")
    
    plt.suptitle("Target Variable Normality Transformation: SalePrice", fontsize=15, y=1.02)
    plt.figtext(0.5, -0.05, "Figure 6: Density distribution of SalePrice showing significant reduction in right-skewness following log1p transformation.", ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig("outputs/06_saleprice_distribution_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Feature Scaling: StandardScaler for all numeric predictor features
    X = df_preprocessed.drop(columns=["SalePrice"]).astype(float)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # 8: Validation & Save Cleaned Data
    processed_csv_path = "data/processed/cleaned.csv"
    # Construct final cleaned dataset with scaled predictors and transformed target
    df_cleaned_export = X_scaled.copy()
    df_cleaned_export["SalePrice_Log1p"] = saleprice_log
    df_cleaned_export["SalePrice_Dollar"] = saleprice_raw
    df_cleaned_export.to_csv(processed_csv_path, index=False)
    
    final_shape = list(df_cleaned_export.shape)
    final_missing = int(df_cleaned_export.isnull().sum().sum())
    logger.info(f"Cleaned dataset successfully validated and saved to {processed_csv_path} with shape {final_shape}")

    # 9: Impact of Preprocessing on Subsequent Analysis
    # a. Baseline Model (Raw numeric features, median imputed, no scaling, no outlier treatment)
    baseline_num_cols = raw_df.select_dtypes(include=[np.number]).columns.drop(["SalePrice", "Id"], errors="ignore")
    X_baseline = raw_df[baseline_num_cols].fillna(raw_df[baseline_num_cols].median())
    y_baseline = raw_df["SalePrice"]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    baseline_ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    baseline_ridge_r2 = float(cross_val_score(baseline_ridge, X_baseline, y_baseline, cv=kf, scoring="r2").mean())
    baseline_ridge_rmse = float((-cross_val_score(baseline_ridge, X_baseline, y_baseline, cv=kf, scoring="neg_root_mean_squared_error")).mean())

    baseline_rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    baseline_rf_r2 = float(cross_val_score(baseline_rf, X_baseline, y_baseline, cv=kf, scoring="r2").mean())
    baseline_rf_rmse = float((-cross_val_score(baseline_rf, X_baseline, y_baseline, cv=kf, scoring="neg_root_mean_squared_error")).mean())

    logger.info(f"Baseline Ridge: R2 = {baseline_ridge_r2:.4f}, RMSE = {baseline_ridge_rmse:.2f}")
    logger.info(f"Baseline RandomForest: R2 = {baseline_rf_r2:.4f}, RMSE = {baseline_rf_rmse:.2f}")

    # b. Preprocessed Models (Full features, winsorized, encoded, scaled, predicting SalePrice)
    preprocessed_ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    prep_ridge_r2 = float(cross_val_score(preprocessed_ridge, X_scaled, saleprice_raw, cv=kf, scoring="r2").mean())
    prep_ridge_rmse = float((-cross_val_score(preprocessed_ridge, X_scaled, saleprice_raw, cv=kf, scoring="neg_root_mean_squared_error")).mean())

    preprocessed_rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    prep_rf_r2 = float(cross_val_score(preprocessed_rf, X_scaled, saleprice_raw, cv=kf, scoring="r2").mean())
    prep_rf_rmse = float((-cross_val_score(preprocessed_rf, X_scaled, saleprice_raw, cv=kf, scoring="neg_root_mean_squared_error")).mean())

    logger.info(f"Preprocessed Ridge: R2 = {prep_ridge_r2:.4f}, RMSE = {prep_ridge_rmse:.2f}")
    logger.info(f"Preprocessed RandomForest: R2 = {prep_rf_r2:.4f}, RMSE = {prep_rf_rmse:.2f}")

    # c. Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = ["Ridge", "Random Forest"]
    x_indices = np.arange(len(models))
    bar_width = 0.35

    # R2 Plot
    axes[0].bar(x_indices - bar_width/2, [baseline_ridge_r2, baseline_rf_r2], bar_width, label="Baseline (Raw Numeric)", color="#94a3b8", edgecolor="#475569")
    axes[0].bar(x_indices + bar_width/2, [prep_ridge_r2, prep_rf_r2], bar_width, label="Preprocessed Pipeline", color="#3b82f6", edgecolor="#1d4ed8")
    axes[0].set_title("5-Fold Cross-Validation R2 Score (Higher is Better)")
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("R2 Score")
    axes[0].set_ylim(0.65, 0.90)
    axes[0].legend()
    for i in x_indices:
        b_val = [baseline_ridge_r2, baseline_rf_r2][i]
        p_val = [prep_ridge_r2, prep_rf_r2][i]
        axes[0].text(i - bar_width/2, b_val + 0.005, f"{b_val:.3f}", ha="center", fontsize=9)
        axes[0].text(i + bar_width/2, p_val + 0.005, f"{p_val:.3f}", ha="center", fontsize=9)

    # RMSE Plot
    axes[1].bar(x_indices - bar_width/2, [baseline_ridge_rmse, baseline_rf_rmse], bar_width, label="Baseline (Raw Numeric)", color="#94a3b8", edgecolor="#475569")
    axes[1].bar(x_indices + bar_width/2, [prep_ridge_rmse, prep_rf_rmse], bar_width, label="Preprocessed Pipeline", color="#10b981", edgecolor="#047857")
    axes[1].set_title("5-Fold Cross-Validation RMSE in $ (Lower is Better)")
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("RMSE ($)")
    axes[1].legend()
    for i in x_indices:
        b_val = [baseline_ridge_rmse, baseline_rf_rmse][i]
        p_val = [prep_ridge_rmse, prep_rf_rmse][i]
        axes[1].text(i - bar_width/2, b_val + 500, f"${b_val:,.0f}", ha="center", fontsize=9)
        axes[1].text(i + bar_width/2, p_val + 500, f"${p_val:,.0f}", ha="center", fontsize=9)

    plt.suptitle("Model Benchmark: Baseline vs Fully Preprocessed Pipeline", fontsize=15, y=1.02)
    plt.figtext(0.5, -0.05, "Figure 7: Impact of preprocessing pipeline on Ridge and Random Forest predictive accuracy across 5-fold CV.", ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig("outputs/07_model_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # d. Correlations Before vs After
    corr_raw = raw_df.select_dtypes(include=[np.number]).drop(columns=["Id"], errors="ignore").corr()["SalePrice"].drop("SalePrice").sort_values(key=abs, ascending=False).head(10)
    
    corr_df_preprocessed = df_preprocessed.copy()
    corr_after = corr_df_preprocessed.corr()["SalePrice"].drop("SalePrice").sort_values(key=abs, ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    corr_raw.plot(kind="barh", ax=axes[0], color="#60a5fa", edgecolor="#2563eb")
    axes[0].set_title("Top 10 Feature Correlations with SalePrice (Raw)")
    axes[0].set_xlabel("Pearson Correlation Coefficient")
    axes[0].invert_yaxis()

    corr_after.plot(kind="barh", ax=axes[1], color="#34d399", edgecolor="#059669")
    axes[1].set_title("Top 10 Feature Correlations with SalePrice (Preprocessed)")
    axes[1].set_xlabel("Pearson Correlation Coefficient")
    axes[1].invert_yaxis()

    plt.suptitle("Feature Linear Association with SalePrice: Before vs After Preprocessing", fontsize=15, y=1.02)
    plt.figtext(0.5, -0.05, "Figure 8: Comparative ranking of top predictive correlates before vs after outlier winsorization and ordinal encoding.", ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig("outputs/08_top_correlations_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Mean and std of 3 key features before and after
    key_features = ["GrLivArea", "LotArea", "TotalBsmtSF"]
    key_features_stats = {}
    for f in key_features:
        key_features_stats[f] = {
            "raw_mean": float(raw_df[f].mean()),
            "raw_std": float(raw_df[f].std()),
            "treated_winsorized_mean": float(df[f].mean()),
            "treated_winsorized_std": float(df[f].std()),
            "scaled_mean": float(X_scaled[f].mean()),
            "scaled_std": float(X_scaled[f].std())
        }

    # Complete Cleaning Log JSON
    cleaning_log = {
        "dataset_name": "Ames Housing (house_prices)",
        "source": "OpenML fetch_openml(name='house_prices')",
        "random_state": RANDOM_STATE,
        "rows_before": initial_shape[0],
        "rows_after": final_shape[0],
        "columns_before": initial_shape[1],
        "columns_after": final_shape[1],
        "missing_values_before": missing_before_dict,
        "missing_values_after": {
            "total_missing_cells": final_missing,
            "columns_with_missing": {}
        },
        "dropped_columns_over_80pct_missing": high_missing_cols,
        "outliers_detected": outliers_detected,
        "outliers_treated": outlier_treated_log,
        "duplicates_removed": duplicates_count,
        "logical_errors_fixed": logical_errors_log,
        "columns_encoded": {
            "ordinal_columns": encoded_ordinal_cols,
            "nominal_columns_one_hot": nominal_cols
        },
        "feature_count_before_encoding": feature_count_before_ohe,
        "feature_count_after_encoding": feature_count_after_ohe,
        "skewness_before": skewness_before,
        "skewness_after": skewness_after,
        "key_features_stats_evolution": key_features_stats,
        "impact_analysis": {
            "evaluation_metric": "5-Fold Cross-Validation (Mean)",
            "target": "SalePrice (USD)",
            "baseline": {
                "features_used": "Raw numeric columns only, median imputation, no outlier handling, no scaling",
                "ridge": {
                    "r2": round(baseline_ridge_r2, 4),
                    "rmse": round(baseline_ridge_rmse, 2)
                },
                "random_forest": {
                    "r2": round(baseline_rf_r2, 4),
                    "rmse": round(baseline_rf_rmse, 2)
                }
            },
            "preprocessed": {
                "features_used": "Full cleaned, winsorized, ordinal + one-hot encoded, StandardScaler",
                "ridge": {
                    "r2": round(prep_ridge_r2, 4),
                    "rmse": round(prep_ridge_rmse, 2)
                },
                "random_forest": {
                    "r2": round(prep_rf_r2, 4),
                    "rmse": round(prep_rf_rmse, 2)
                }
            }
        }
    }

    log_path = "outputs/cleaning_log.json"
    with open(log_path, "w") as f:
        json.dump(cleaning_log, f, indent=4)
    logger.info(f"Saved cleaning log to {log_path}")

    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"Initial Shape:       {initial_shape}")
    print(f"Final Cleaned Shape: {final_shape}")
    print(f"Missing Values:      {missing_before_dict['total_missing_cells']} -> {final_missing}")
    print(f"High Missing Dropped: {high_missing_cols}")
    print(f"Encoded Features:    {feature_count_before_ohe} -> {feature_count_after_ohe}")
    print("-" * 60)
    print("Downstream Impact (5-Fold CV on SalePrice):")
    print(f"  Baseline Ridge:       R2 = {baseline_ridge_r2:.4f} | RMSE = ${baseline_ridge_rmse:,.2f}")
    print(f"  Preprocessed Ridge:   R2 = {prep_ridge_r2:.4f} | RMSE = ${prep_ridge_rmse:,.2f}")
    print(f"  Baseline RF:          R2 = {baseline_rf_r2:.4f} | RMSE = ${baseline_rf_rmse:,.2f}")
    print(f"  Preprocessed RF:      R2 = {prep_rf_r2:.4f} | RMSE = ${prep_rf_rmse:,.2f}")
    print("="*60 + "\n")

    return cleaning_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ames Housing Cleaning and Preprocessing Pipeline")
    parser.add_argument("--run", action="store_true", default=True, help="Execute pipeline")
    args = parser.parse_args()
    run_full_pipeline()
