# Week 2: Exploratory Data Analysis (EDA) and Visualization

## Overview
This module performs a comprehensive Exploratory Data Analysis on the **Bike Sharing Demand** dataset. The goal is to understand demand patterns, uncover correlations, detect anomalies, and derive actionable insights.

## Dataset
- **Source:** UCI Machine Learning Repository — Bike Sharing Dataset
- **Loaded via:** `sklearn.datasets.fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)` (fallback: UCI `hour.csv`)
- **Records:** ~17,379 hourly observations
- **Features:** Season, year, month, hour, holiday, weekday, workingday, weather, temperature, humidity, windspeed, and bike count

## Folder Structure
```
Week2/
├── data/
│   ├── raw/raw.csv                  # Untouched original dataset
│   └── processed/eda_ready.csv      # Cleaned & feature-engineered dataset
├── notebooks/
│   └── week2_eda.ipynb              # Main deliverable (full EDA notebook)
├── src/
│   └── eda.py                       # Reusable EDA pipeline script
├── outputs/
│   ├── figures/                     # All plots as PNG (fig01_... naming)
│   ├── tables/                      # Summary tables as CSV
│   ├── eda_summary.json             # Key statistics and findings
│   └── key_findings.md              # Plain-English top insights
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
cd Week2
pip install -r requirements.txt
```

### 2. Run the reusable script
```bash
cd Week2
python src/eda.py
```

### 3. Open the Notebook
```bash
cd Week2
jupyter notebook notebooks/week2_eda.ipynb
```
Run all cells from top to bottom (Kernel → Restart & Run All).

## Outputs
| Output | Description |
|--------|-------------|
| `outputs/figures/fig01_*.png` to `fig16_*.png` | All EDA plots |
| `outputs/tables/describe_numeric.csv` | Numeric summary statistics |
| `outputs/tables/describe_categorical.csv` | Categorical frequency tables |
| `outputs/tables/correlation_matrix.csv` | Full correlation matrix |
| `outputs/tables/avg_count_by_*.csv` | Groupby aggregations |
| `outputs/eda_summary.json` | Machine-readable key findings |
| `outputs/key_findings.md` | Human-readable insights |

## Key Findings (Preview)
- Peak demand hours: **8 AM and 5–6 PM** (commute-driven)
- Busiest season: **Fall**, slowest: **Spring**
- Rain/snow reduces average demand by **~50–60%**
- Year-over-year growth: **~60–65% increase** from 2011 to 2012
- Strongest correlations: `temp` (0.40), `feel_temp` (0.39)

## Author
Part of the *Virtual Data Science with Python* Internship — Week 2
