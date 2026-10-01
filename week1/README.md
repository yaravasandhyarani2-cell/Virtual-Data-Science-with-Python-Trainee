# Week 1: Data Acquisition, Cleaning, and Preprocessing
**Project:** Ames Housing Price Prediction & Feature Engineering Pipeline  
**Dataset:** Ames Housing (`sklearn.datasets.fetch_openml(name="house_prices", as_frame=True)`)  
**Repository:** YuvaIntern_Data_Science / Week1  

---

## 1. Overview & Dataset

This directory contains the complete deliverable for **Week 1: Data Acquisition, Cleaning, and Preprocessing**. The Ames Housing dataset captures residential real estate transactions in Ames, Iowa (2006–2010), compiled by Dean De Cock from official Ames City Assessor's records. It contains 1,460 observations across 80 physical, architectural, and geographic characteristics alongside the target `SalePrice`.

The goal of this week's task is to build a robust, reproducible data acquisition, cleaning, preprocessing, and auditing pipeline that eliminates all data quality defects, handles high-leverage outliers, normalizes target skewness, and empirically validates the downstream impact on predictive regression models (Ridge and Random Forest).

---

## 2. Folder Structure

All Week 1 assets are strictly self-contained within the `Week1` directory:

```text
Week1/
├── data/
│   ├── raw/
│   │   └── raw.csv                   # Untouched original OpenML download (1,460 x 81)
│   └── processed/
│       └── cleaned.csv               # Validated preprocessed dataset (1,460 x 233)
├── notebooks/
│   └── week1_cleaning.ipynb          # Master executed Jupyter Notebook (10 numbered sections)
├── src/
│   └── clean.py                      # Production reusable cleaning & preprocessing CLI script
├── outputs/
│   ├── 01_missing_values.png         # Missing percentage bar plot with 80% threshold line
│   ├── 02_correlation_heatmap_raw.png # Correlation heatmap of top numeric features
│   ├── 03_raw_distributions.png      # Skewness and distribution curves for key area & price features
│   ├── 04_outliers_boxplot_before_after.png  # Comparative box plots before vs after winsorization
│   ├── 05_outliers_scatter_before_after.png  # Bivariate scatter plots vs SalePrice before vs after
│   ├── 06_saleprice_distribution_before_after.png # Target normality: Raw vs log1p SalePrice
│   ├── 07_model_performance_comparison.png       # 5-fold CV R2 and RMSE bar charts
│   ├── 08_top_correlations_before_after.png      # Top 10 linear correlates before vs after
│   └── cleaning_log.json             # Exhaustive audit log documenting all transforms and CV metrics
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules for Week 1
└── README.md                         # This documentation file
```

---

## 3. How to Install and Run

All commands below assume you are operating from **inside the `Week1` folder**:

```bash
# Navigate to Week1 directory
cd Week1
```

### 3.1 Install Dependencies
Install the required packages using Python (virtual environment recommended):
```bash
# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment:
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3.2 Run the Reusable Python Pipeline
Execute `src/clean.py` to acquire data, clean, transform, validate, run 5-fold CV benchmarks, and regenerate all outputs:
```bash
python src/clean.py
```
*Expected terminal summary:*
```text
============================================================
PIPELINE EXECUTION SUMMARY
============================================================
Initial Shape:       [1460, 81]
Final Cleaned Shape: [1460, 233]
Missing Values:      6965 -> 0
High Missing Dropped: ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
Encoded Features:    75 -> 231
------------------------------------------------------------
Downstream Impact (5-Fold CV on SalePrice):
  Baseline Ridge:       R2 = 0.7438 | RMSE = $37,793.98
  Preprocessed Ridge:   R2 = 0.8438 | RMSE = $28,074.77
  Baseline RF:          R2 = 0.8362 | RMSE = $30,636.32
  Preprocessed RF:      R2 = 0.8772 | RMSE = $25,293.06
============================================================
```

### 3.3 Run the Jupyter Notebook
Launch the notebook to inspect interactive visualizations, detailed commentary ("What I'm doing", "Why (rationale)", "Alternative considered", and "Observation" cells), and outputs:
```bash
jupyter notebook notebooks/week1_cleaning.ipynb
```
Inside Jupyter, click **Kernel -> Restart & Run All**. The notebook runs top to bottom cleanly and automatically regenerates `data/raw/raw.csv`, `data/processed/cleaned.csv`, `outputs/cleaning_log.json`, and all plot PNGs in `outputs/`.

---

## 4. Key Pipeline Methodology

1. **Initial Exploration:** Audits 81 features (38 numeric, 43 categorical). Identifies 19 features with missing values totaling 6,965 cells.
2. **Missing Values Treatment:**
   - **80% Rule:** Pruned `PoolQC` (99.52%), `MiscFeature` (96.30%), `Alley` (93.77%), `Fence` (80.75%) due to severe sparsity and negligible variance.
   - **Structural NaNs:** Imputed `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `MasVnrType` with `"None"`.
   - **Contextual Grouped Median:** `LotFrontage` (259 missing) imputed with the median of each respective `Neighborhood`.
   - **General:** Remaining numerics imputed with column medians; categoricals with modes. Total missing cells: **0**.
3. **Outlier Treatment (Winsorization at 1st & 99th Percentile):**
   - IQR and Z-score detection flagged extreme values in `GrLivArea`, `LotArea`, `TotalBsmtSF`, and `SalePrice`.
   - Capped extreme tails at 1st and 99th percentiles rather than dropping rows. This preserves 100% of the 1,460 observations and prevents sample shrinkage while controlling gradient leverage.
4. **Erroneous Entries Fixed:**
   - Duplicates verified (0).
   - Timeline consistency asserted: `YearRemodAdd >= YearBuilt` and `YrSold >= YearBuilt`.
   - Area measurements verified non-negative (0 negative values).
   - Standardized string whitespace and casing.
   - Corrected semantic data type of `MSSubClass` from integer to categorical string. Dropped `Id`.
5. **Preprocessing & Encodings:**
   - **Ordinal Encoding:** 9 quality ratings mapped to `None=0 < Po=1 < Fa=2 < TA=3 < Gd=4 < Ex=5`.
   - **One-Hot Encoding:** Nominal categoricals converted via `pd.get_dummies(..., drop_first=True)` (expanding predictors to 231).
   - **Target log1p:** Compressed `SalePrice` skewness from **1.27 to 0.16** (near-Gaussian normality).
   - **StandardScaler:** Scaled predictors to mean $\approx 0$ and variance $\approx 1$.
6. **Validation:** Asserted zero missing values across the 1,460 × 233 cleaned matrix; persisted to `data/processed/cleaned.csv`.

---

## 5. Downstream Modeling Impact (5-Fold CV on SalePrice)

| Model | Pipeline Configuration | Mean $R^2$ Score | Mean RMSE ($) | Error Reduction ($) | % Error Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | Baseline (Raw Numeric) | 0.7438 | $37,794 | — | — |
| **Ridge Regression** | **Preprocessed Pipeline** | **0.8438** | **$28,075** | **-$9,719** | **-25.7%** |
| **Random Forest** | Baseline (Raw Numeric) | 0.8362 | $30,636 | — | — |
| **Random Forest** | **Preprocessed Pipeline** | **0.8772** | **$25,293** | **-$5,343** | **-17.4%** |

---

## 6. Generated Output Artifacts

All plots and logs are saved inside `outputs/`:
- `outputs/01_missing_values.png`
- `outputs/02_correlation_heatmap_raw.png`
- `outputs/03_raw_distributions.png`
- `outputs/04_outliers_boxplot_before_after.png`
- `outputs/05_outliers_scatter_before_after.png`
- `outputs/06_saleprice_distribution_before_after.png`
- `outputs/07_model_performance_comparison.png`
- `outputs/08_top_correlations_before_after.png`
- `outputs/cleaning_log.json`
