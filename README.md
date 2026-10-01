# YuvaIntern_Data_Science
Repository for Data Science and Machine Learning Internship Tasks.

## Repository Architecture

This repository is organized strictly by weekly task modules. Each week lives in its own dedicated, self-contained directory with its own data, source code, notebooks, documentation, and outputs:

```text
YuvaIntern_Data_Science/
├── Week1/                            # Week 1: Data Acquisition, Cleaning, and Preprocessing
│   ├── data/
│   │   ├── raw/raw.csv
│   │   └── processed/cleaned.csv
│   ├── notebooks/
│   │   └── week1_cleaning.ipynb
│   ├── src/
│   │   └── clean.py
│   ├── outputs/
│   │   ├── (all plots as PNG)
│   │   └── cleaning_log.json
│   ├── requirements.txt
│   ├── README.md
│   └── .gitignore
├── Week2/                            # Week 2: Exploratory Data Analysis and Visualization
│   ├── data/
│   │   ├── raw/raw.csv
│   │   └── processed/eda_ready.csv
│   ├── notebooks/
│   │   └── week2_eda.ipynb
│   ├── src/
│   │   └── eda.py
│   ├── outputs/
│   │   ├── figures/ (all 16+ plots as PNG)
│   │   ├── tables/ (all CSV summary tables)
│   │   ├── eda_summary.json
│   │   └── key_findings.md
│   ├── requirements.txt
│   └── README.md
└── ...
```

## Policy for All Future Weeks
- Every new weekly task goes into its own dedicated sibling folder (`Week2`, `Week3`, etc.).
- Never mix files across different weeks.
- All code, notebooks, datasets, dependencies, and outputs remain strictly inside the respective week's folder.

## Completed Modules
- [Week 1: Data Acquisition, Cleaning, and Preprocessing](./Week1/README.md)
- [Week 2: Exploratory Data Analysis (EDA) and Visualization](./Week2/README.md)
