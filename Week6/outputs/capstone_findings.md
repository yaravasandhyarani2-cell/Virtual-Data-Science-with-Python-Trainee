# Week 6: Final Capstone Project - Integrative Data Science Pipeline

## 1. Executive Summary
This Capstone project synthesizes all foundational modules (EDA, Data Cleaning, Unsupervised Clustering, and Supervised Machine Learning) into a cohesive, production-grade end-to-end Python pipeline using the **California Housing** dataset.

## 2. Pipeline Stages
1. **Data Ingestion:** Loaded raw housing data (20,640 records, 8 numeric features, 1 target variable `MedHouseVal`).
2. **Preprocessing & Feature Engineering:**
   - Engineered domain features: `Rooms_Per_Bed`, `Bedrms_Per_Person`, and geospatial coordinate interaction.
   - Outlier clipping applied to extreme ratios at 99th percentile.
   - Standardized features with `StandardScaler` to prevent leakage.
3. **Unsupervised Market Segmentation:**
   - K-Means algorithm grouped properties into 4 distinct geographic-economic clusters with a Silhouette score of **0.4444**.
4. **Supervised Regression Modeling:**
   - Evaluated `Random Forest Regressor` and `Gradient Boosting Regressor` with 5-fold cross-validation.

## 3. Supervised Model Evaluation Summary

| Model | 5-Fold CV R² | 5-Fold CV RMSE | Test R² | Test RMSE ($100k) | Test MAE ($100k) |
|---|---|---|---|---|---|
| **Random Forest Regressor** | 0.7934 | 0.5254 | **0.7918** | **0.5224** | **0.3465** |
| **Gradient Boosting Regressor** | 0.8218 | 0.4881 | 0.8185 | 0.4876 | 0.3261 |

## 4. Key Findings & Insights
- **Top Predictive Driver:** `MedInc` (Median Income) accounts for over 50% of model importance, proving that local earning power dominates property values.
- **Geographic Clusters:** Coastal and urban clusters (Bay Area and Los Angeles) exhibit significantly higher median home values compared to inland clusters.
- **Champion Model:** **Gradient Boosting Regressor** demonstrated exceptional predictive performance with a Test R² of **0.8185** and an RMSE of **0.4876**.
