# Week 6: Final Capstone Project - Integrative Data Science Pipeline

## Overview
This Capstone project synthesizes all data science internship competencies—**Exploratory Data Analysis, Data Cleaning & Preprocessing, Unsupervised Clustering, and Supervised Predictive Modeling**—into a single modular end-to-end Python pipeline using the **California Housing** public benchmark dataset.

---

## Directory Structure
```text
Week6/
├── data/
│   ├── raw/
│   │   └── raw.csv                          # Untouched raw dataset (20,640 records x 9 columns)
│   └── processed/
│       ├── train_processed.csv              # Scaled training set with engineered features
│       ├── test_processed.csv               # Scaled test set with engineered features
│       ├── scaler.joblib                    # Serialized StandardScaler
│       ├── cluster_profiles.csv             # K-Means regional market cluster profiles
│       └── model_performance_summary.csv    # Supervised models CV & test evaluation table
├── figures/
│   ├── fig01_correlation_heatmap.png        # Complete feature correlation matrix
│   ├── fig02_geospatial_clusters.png        # Geospatial K-Means market segmentation scatter map
│   ├── fig03_actual_vs_predicted.png        # Actual vs Predicted regression fits
│   └── fig04_feature_importances.png        # Random Forest relative feature importances
├── models/
│   ├── random_forest_regressor.joblib       # Serialized Random Forest model
│   └── gradient_boosting_regressor.joblib   # Serialized Gradient Boosting model
├── outputs/
│   ├── capstone_summary.json                # Complete machine-readable pipeline summary
│   └── capstone_findings.md                 # Detailed executive report and data insights
├── src/
│   └── capstone_pipeline.py                 # Fully automated, modular end-to-end pipeline
└── requirements.txt                         # Package dependencies
```

---

## Integrative Pipeline Architecture

1. **Data Ingestion & Integrity Checks**:
   - Acquired California Housing dataset (20,640 block groups, 8 numeric predictive attributes, target: `MedHouseVal`).
   - Verified zero missing values across all records.

2. **Cleaning & Feature Preprocessing**:
   - **Feature Engineering**: Formulated domain attributes:
     - `Rooms_Per_Bed = AveRooms / AveBedrms`
     - `Bedrms_Per_Person = AveBedrms / AveOccup`
     - `Coord_Interaction = Latitude * Longitude`
   - **Outlier Handling**: Clipped extreme values at the 99th percentile for room/occupancy ratios.
   - **Feature Normalization**: Standardized features using `StandardScaler` fitted strictly on train partition.

3. **Unsupervised Market Segmentation (Clustering)**:
   - Applied **K-Means ($k=4$)** on coordinates (`Latitude`, `Longitude`) and socioeconomic level (`MedInc`).
   - Segments cleanly isolate Bay Area high-value, Southern California urban, Central Valley agricultural, and Northern coastal regions.

4. **Supervised Regression Modeling & Cross-Validation**:
   - Models evaluated with 5-fold cross-validation:
     - **Random Forest Regressor** (100 estimators, max depth 12)
     - **Gradient Boosting Regressor** (100 estimators, max depth 5, learning rate 0.1)

---

## How to Run the Pipeline

From the project root:
```powershell
python Week6/src/capstone_pipeline.py
```

Or from inside `Week6/`:
```powershell
cd Week6
python src/capstone_pipeline.py
```
All outputs, models, CSV profiles, and visualization figures will be generated and saved automatically.
