# Ames Housing: Data Acquisition, Cleaning, and Preprocessing Pipeline
*Week 1: Foundations of Machine Learning & Applied Data Engineering*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Pipeline-Verified%20%26%20Reproducible-brightgreen.svg)](#)

---

## 1. Project Overview & Data Source

This project provides an end-to-end, production-grade data acquisition, cleaning, preprocessing, and validation pipeline built on the **Ames Housing dataset**. Raw real estate transaction records typically suffer from structural missingness (features absent in a property), extreme physical leverage outliers (multi-acre agricultural parcels, massive estates), heavy target right-skewness, and semantic datatype misclassifications. 

This repository standardizes and remediates all data anomalies, producing clean, validated outputs and demonstrating a significant empirical improvement in downstream predictive modeling.

### Data Source and Collection Method
- **Origin:** Compiled by **Dean De Cock** (Truman State University) and published in the *Journal of Statistics Education* (Vol. 19, No. 3, 2011) as a modern, high-dimensional successor to the Boston Housing dataset.
- **Collection Method:** Direct administrative extraction from the **Ames City Assessor's Office** in Ames, Iowa, covering real residential sales transactions between **2006 and 2010**.
- **Acquisition Protocol:** Programmatically acquired via Scikit-Learn OpenML integration (`sklearn.datasets.fetch_openml(name="house_prices", as_frame=True)`), yielding 1,460 residential properties across 80 physical, zoning, and architectural characteristics plus nominal `SalePrice`.

---

## 2. Repository Structure

```text
├── data/
│   ├── raw/
│   │   └── raw.csv                   # Untouched, pristine original dataset from OpenML (1,460 x 81)
│   └── processed/
│       └── cleaned.csv               # Fully preprocessed, validated dataset (1,460 x 233)
├── notebooks/
│   └── week1_cleaning.ipynb          # Master deliverable: fully executed notebook with 10 sections & commentary
├── src/
│   └── clean.py                      # Reusable, automated production cleaning and preprocessing CLI pipeline
├── outputs/
│   ├── 01_missing_values.png         # Bar plot of missing value counts & percentages with 80% threshold line
│   ├── 02_correlation_heatmap_raw.png # Correlation heatmap of top numeric features associated with SalePrice
│   ├── 03_raw_distributions.png      # Histograms & KDE distributions showing raw feature skewness
│   ├── 04_outliers_boxplot_before_after.png  # Comparative box plots before vs after winsorization
│   ├── 05_outliers_scatter_before_after.png  # Bivariate scatter plots vs SalePrice before vs after winsorization
│   ├── 06_saleprice_distribution_before_after.png # Normality comparison: Raw vs log1p SalePrice distribution
│   ├── 07_model_performance_comparison.png       # 5-fold CV R2 and RMSE bar charts (Baseline vs Preprocessed)
│   ├── 08_top_correlations_before_after.png      # Top 10 feature linear correlates with SalePrice before vs after
│   └── cleaning_log.json             # Exhaustive audit log documenting all transforms and CV metrics
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules for virtualenvs, caches, and checkpoints
└── README.md                         # Comprehensive documentation and execution guide
```

---

## 3. Key Pipeline Stages

### Section 1: Introduction and Data Source
Establishes the empirical context, research methodology, and municipal data collection protocols of the Ames Assessor's records.

### Section 2: Data Acquisition
Acquires the dataset directly from OpenML and writes an untouched baseline copy to `data/raw/raw.csv` to ensure data provenance and reproducibility.

### Section 3: Initial Exploration & Structural Auditing
- Audits feature cardinality, types (38 numeric, 43 categorical), and basic statistics.
- Identifies 19 features containing missing values (6,965 total missing cells).
- Analyzes Pearson correlations and identifies high collinearity among physical metrics (e.g. `GarageCars` and `GarageArea`, r = 0.88).
- Measures skewness across key metrics: `LotArea` (12.21), `SalePrice` (1.88), `GrLivArea` (1.37), and `TotalBsmtSF` (1.52).

### Section 4: Domain-Guided Missing Value Imputation
- **80% Drop Rule:** Four columns (`PoolQC` 99.52%, `MiscFeature` 96.30%, `Alley` 93.77%, `Fence` 80.75%) exhibit extreme sparsity and lack variance; dropped with domain justification.
- **Structural "Feature Absent" Imputation:** Features where NaN signifies absence of an amenity (`FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `MasVnrType`) are imputed with `"None"`.
- **Contextual Grouped Median:** `LotFrontage` (259 missing values) is imputed with the median of its specific suburban `Neighborhood`.
- **General Fallbacks:** Remaining numerical columns are imputed with column medians; categorical columns with modes. Missing cells: **6,965 -> 0**.

### Section 5: Outlier Auditing and Statistical Winsorization
- **Detection:** Evaluated via IQR (1.5 × IQR) and Z-score (|z| > 3) thresholds across `GrLivArea`, `LotArea`, `TotalBsmtSF`, and `SalePrice`.
- **Treatment Strategy (Winsorization at 1st & 99th Percentiles):**
  * *Why Winsorization over Row Deletion?* Deleting all flagged IQR observations would eliminate >100 rows (>7% of the dataset), discarding valid luxury homes and architectural diversity. Winsorization caps extreme tail leverage (e.g., `LotArea` capped at 37,568 sq ft, `GrLivArea` at 3,123 sq ft, `TotalBsmtSF` at 2,155 sq ft) without shrinking sample size.

### Section 6: Erroneous Entries & Semantic Types
- Audits duplicate rows (0 found).
- Validates chronological sanity: `YearRemodAdd >= YearBuilt` and `YrSold >= YearBuilt`.
- Audits square footage and acreage for negative values (0 found).
- Trims whitespace across all string features.
- Corrects semantic datatype of `MSSubClass` from integer to string categorical (preventing spurious numeric distance assumptions).
- Drops arbitrary primary key `Id`.

### Section 7: Preprocessing & Feature Engineering
- **Ordinal Encoding:** Standardizes 9 quality/condition features (`ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`) using the rating scale:  
  `None=0 < Po=1 < Fa=2 < TA=3 < Gd=4 < Ex=5`.
- **One-Hot Encoding:** Applies `pd.get_dummies(..., drop_first=True)` to all remaining nominal variables, expanding feature space from 75 to 231 predictors.
- **Target Transformation:** Applies `log1p` transformation to `SalePrice`, compressing skewness from **1.27 down to 0.16** (near-Gaussian normality).
- **StandardScaler:** Normalizes all predictor features to mean ~ 0 and standard deviation ~ 1.

### Section 8: Validation & Persistence
- Verifies zero missing values, finite real numbers, and consistent row dimensions.
- Exports validated dataset to `data/processed/cleaned.csv` containing scaled predictors, `SalePrice_Dollar`, and `SalePrice_Log1p`.

### Section 9: Downstream Predictive Impact Analysis
To quantify the value of preprocessing, two models (**Ridge Regression** and **Random Forest Regressor**) were trained under standardized **5-fold cross-validation** predicting `SalePrice`:
- **Baseline:** Raw numeric columns only, simple median imputation, unscaled, no outlier treatment.
- **Preprocessed:** Fully cleaned, winsorized, ordinally and one-hot encoded, and standardized.

#### Empirical Benchmark Results

| Model | Pipeline Configuration | Mean $R^2$ Score | Mean RMSE ($) | RMSE Reduction ($) | % Error Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | Baseline (Raw Numeric) | 0.7438 | $37,794 | — | — |
| **Ridge Regression** | **Preprocessed Pipeline** | **0.8438** | **$28,075** | **-$9,719** | **-25.7%** |
| **Random Forest** | Baseline (Raw Numeric) | 0.8362 | $30,636 | — | — |
| **Random Forest** | **Preprocessed Pipeline** | **0.8772** | **$25,293** | **-$5,343** | **-17.4%** |

### Section 10: Challenges Encountered & Real-World Solutions
Detailed markdown retrospective covering structural absence vs. random missingness, high-missingness threshold justification, contextual grouped imputation, outlier leverage mitigation, and matrix condition optimization.

---

## 4. How to Run

### Prerequisites
- Python 3.10 or higher
- Git

### Step 1: Clone the Repository & Setup Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Run the Automated Preprocessing Pipeline
Execute the reusable pipeline script to fetch data, clean, transform, validate, benchmark models, and save all outputs:
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

### Step 3: Run the Jupyter Notebook
To inspect interactive figures, commentary cells, and step-by-step code:
```bash
jupyter notebook notebooks/week1_cleaning.ipynb
```
Select `Kernel -> Restart & Run All` to verify end-to-end execution.

---

## 5. Output Artifacts

All figures and logs are automatically generated and saved to the `outputs/` directory:
- `outputs/01_missing_values.png`: Missing percentages per column with 80% threshold line.
- `outputs/02_correlation_heatmap_raw.png`: Pearson correlation matrix of top numeric features.
- `outputs/03_raw_distributions.png`: Skewness and distribution curves for key area and price features.
- `outputs/04_outliers_boxplot_before_after.png`: Comparative box plots showing winsorization effects.
- `outputs/05_outliers_scatter_before_after.png`: Bivariate scatter plots highlighting leverage control.
- `outputs/06_saleprice_distribution_before_after.png`: Target normality transformation (skewness reduced from 1.27 to 0.16).
- `outputs/07_model_performance_comparison.png`: Grouped bar charts of $R^2$ and RMSE for Ridge and Random Forest.
- `outputs/08_top_correlations_before_after.png`: Top 10 correlates with SalePrice before vs after preprocessing.
- `outputs/cleaning_log.json`: Machine-readable audit file containing row/column counts, missingness stats, outlier parameters, encoding maps, and benchmark metrics.

---

## 6. License
This project is open-source under the MIT License.
