"""
Week 2: Exploratory Data Analysis (EDA) and Visualization
Dataset: Bike Sharing Demand (UCI / OpenML)
Author: Virtual Data Science with Python Internship
"""

# ---------------------------------------------
# 0. SETUP - ensure working directory is Week2
# ---------------------------------------------
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

# Locate the Week2 root dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))   # .../Week2/src
week2_root = os.path.dirname(script_dir)                  # .../Week2
os.chdir(week2_root)
print(f"[INFO] Working directory set to: {os.getcwd()}")

# ---------------------------------------------
# 1. IMPORTS
# ---------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import zscore

# Global style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})
PALETTE = "Set2"
RANDOM_STATE = 42

# ---------------------------------------------
# DIRECTORIES
# ---------------------------------------------
RAW_PATH       = os.path.join("data", "raw", "raw.csv")
PROC_PATH      = os.path.join("data", "processed", "eda_ready.csv")
FIG_DIR        = os.path.join("outputs", "figures")
TABLE_DIR      = os.path.join("outputs", "tables")
SUMMARY_JSON   = os.path.join("outputs", "eda_summary.json")
FINDINGS_MD    = os.path.join("outputs", "key_findings.md")

for d in [os.path.dirname(RAW_PATH), os.path.dirname(PROC_PATH),
          FIG_DIR, TABLE_DIR, os.path.dirname(SUMMARY_JSON)]:
    os.makedirs(d, exist_ok=True)

fig_counter = [0]   # mutable counter shared across helpers

def save_fig(name: str) -> str:
    """Save current figure with sequential numbering."""
    fig_counter[0] += 1
    fname = f"fig{fig_counter[0]:02d}_{name}.png"
    fpath = os.path.join(FIG_DIR, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fpath}")
    return fpath


# =============================================================================
# SECTION 1 - DATA ACQUISITION
# =============================================================================
def acquire_data() -> pd.DataFrame:
    """Download Bike Sharing Demand and save raw CSV."""
    print("\n" + "="*60)
    print("SECTION 1 - DATA ACQUISITION")
    print("="*60)

    if os.path.exists(RAW_PATH):
        print(f"[INFO] Raw data already exists at {RAW_PATH}. Loading ...")
        df = pd.read_csv(RAW_PATH)
        return df

    # Try OpenML first
    try:
        from sklearn.datasets import fetch_openml
        print("[INFO] Fetching from OpenML (Bike_Sharing_Demand v2) ...")
        bunch = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True,
                              parser="auto")
        df = bunch.frame
        source = "OpenML (sklearn.datasets.fetch_openml)"
    except Exception as e:
        print(f"[WARN] OpenML failed ({e}). Falling back to UCI direct download ...")
        import urllib.request
        import zipfile, io
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "00275/Bike-Sharing-Dataset.zip")
        print(f"[INFO] Downloading from {url} ...")
        with urllib.request.urlopen(url, timeout=60) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
        with z.open("hour.csv") as f:
            df = pd.read_csv(f)
        source = "UCI ML Repository (hour.csv fallback)"

    print(f"[INFO] Data source: {source}")
    df.to_csv(RAW_PATH, index=False)
    print(f"[INFO] Raw data saved -> {RAW_PATH}")
    print(f"[INFO] Shape: {df.shape}")
    return df


# =============================================================================
# SECTION 2 - INITIAL ANALYSIS
# =============================================================================
def initial_analysis(df: pd.DataFrame) -> dict:
    """Shape, dtypes, head/tail, describe, missing, duplicates, skew, kurtosis."""
    print("\n" + "="*60)
    print("SECTION 2 - INITIAL ANALYSIS")
    print("="*60)

    info = {}
    info["rows"], info["columns"] = df.shape
    print(f"Shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")
    print(f"\nMissing values:\n{df.isnull().sum().to_string()}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(exclude=np.number).columns.tolist()

    desc_num = df[numeric_cols].describe().T
    desc_num["skewness"] = df[numeric_cols].skew()
    desc_num["kurtosis"] = df[numeric_cols].kurt()
    desc_num.to_csv(os.path.join(TABLE_DIR, "describe_numeric.csv"))
    print(f"\nNumeric describe (saved to tables):\n{desc_num.to_string()}")

    info["missing_values"]  = df.isnull().sum().to_dict()
    info["duplicates"]      = int(df.duplicated().sum())
    info["numeric_columns"] = numeric_cols
    info["categorical_columns"] = cat_cols

    return info


# =============================================================================
# SECTION 3 - DATA PREPARATION FOR EDA
# =============================================================================
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering and label encoding for EDA."""
    print("\n" + "="*60)
    print("SECTION 3 - DATA PREPARATION FOR EDA")
    print("="*60)

    df = df.copy()

    # Standardise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename OpenML columns to unified names
    col_renames = {
        "feel_temp": "atemp",
        "count":     "cnt",
        "hour":      "hr",
        "month":     "mnth",
        "humidity":  "hum",
        "weather":   "weathersit",
    }
    for old, new in col_renames.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Cast numeric columns (exclude season and weathersit which might be string categories)
    num_cols = ["temp", "atemp", "hum", "windspeed", "cnt",
                "casual", "registered", "weekday", "hr", "mnth"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Boolean columns: OpenML stores as Categorical with "True"/"False" strings
    def to_bool_int(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip().str.lower()
        return s.map({"true": 1, "false": 0, "1": 1, "0": 0,
                      "1.0": 1, "0.0": 0}).fillna(0).astype(int)

    for bool_col in ["holiday", "workingday"]:
        if bool_col in df.columns:
            df[bool_col] = to_bool_int(df[bool_col])

    # Readable labels & integer representations
    # Support both string labels (OpenML) and numeric codes 1-4 (UCI)
    season_str_map = {"spring": "Spring", "summer": "Summer", "fall": "Fall", "winter": "Winter"}
    season_num_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    season_to_int  = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

    if "season" in df.columns:
        s_str = df["season"].astype(str).str.strip().str.lower()
        if s_str.isin(season_str_map.keys()).any():
            df["season"] = s_str.map(season_str_map)
            df["season_int"] = df["season"].map(season_to_int).astype("Int64")
        else:
            df["season_int"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
            df["season"] = df["season_int"].map(season_num_map)

    weather_str_map = {
        "clear": "Clear", "misty": "Cloudy", "mist": "Cloudy",
        "rain": "Light Rain/Snow", "light_rain": "Light Rain/Snow",
        "heavy_rain": "Heavy Rain/Snow", "heavy rain": "Heavy Rain/Snow"
    }
    weather_num_map = {1: "Clear", 2: "Cloudy", 3: "Light Rain/Snow", 4: "Heavy Rain/Snow"}
    weather_to_int  = {"Clear": 1, "Cloudy": 2, "Light Rain/Snow": 3, "Heavy Rain/Snow": 4}

    w_col = "weathersit" if "weathersit" in df.columns else ("weather" if "weather" in df.columns else None)
    if w_col:
        w_str = df[w_col].astype(str).str.strip().str.lower()
        if w_str.isin(weather_str_map.keys()).any():
            df["weather_label"] = w_str.map(weather_str_map)
            df["weathersit_int"] = df["weather_label"].map(weather_to_int).astype("Int64")
            df["weathersit"] = df["weathersit_int"]
        else:
            df["weathersit_int"] = pd.to_numeric(df[w_col], errors="coerce").astype("Int64")
            df["weather_label"]  = df["weathersit_int"].map(weather_num_map)
            df["weathersit"] = df["weathersit_int"]

    weekday_map = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed",
                   4: "Thu", 5: "Fri", 6: "Sat"}
    if "weekday" in df.columns:
        df["weekday_int"]   = pd.to_numeric(df["weekday"], errors="coerce").astype("Int64")
        df["weekday_label"] = df["weekday_int"].map(weekday_map)

    # Year: OpenML uses 0/1 int (year col), UCI uses yr col
    yr_col = "yr" if "yr" in df.columns else ("year" if "year" in df.columns else None)
    if yr_col:
        df["yr_int"] = pd.to_numeric(df[yr_col], errors="coerce").astype("Int64")
        df["year"]   = df["yr_int"].map({0: 2011, 1: 2012})

    # Datetime feature
    if "dteday" in df.columns:
        df["datetime"]    = pd.to_datetime(df["dteday"], errors="coerce")
        df["day_of_year"] = df["datetime"].dt.dayofyear

    # Ensure hr and mnth are Int64
    for icol in ["hr", "mnth"]:
        if icol in df.columns:
            df[icol] = df[icol].astype("Int64")

    # day_type (holiday/workingday already cast to int above)
    def get_day_type(row):
        try:
            if int(row.get("holiday", 0)) == 1:
                return "Holiday"
            if int(row.get("workingday", 0)) == 1:
                return "Working Day"
        except (ValueError, TypeError):
            pass
        return "Weekend"

    df["day_type"] = df.apply(get_day_type, axis=1)

    # -- time_of_day bins -----------------------------------------------------
    def time_of_day(hr):
        if pd.isna(hr):
            return "Unknown"
        hr = int(hr)
        if   0  <= hr < 6:   return "Night"
        elif 6  <= hr < 12:  return "Morning"
        elif 12 <= hr < 17:  return "Afternoon"
        else:                return "Evening"

    if "hr" in df.columns:
        df["time_of_day"] = df["hr"].apply(time_of_day)

    # Temperature bins (OpenML temp is in Celsius ~0-41, UCI is normalized 0-1)
    if "temp" in df.columns:
        t = df["temp"].astype(float)
        t_max = t.max()
        if t_max > 2.0:   # Celsius
            bins   = [t.min()-1, 10, 20, 30, t_max+1]
            labels = ["Cold", "Mild", "Warm", "Hot"]
        else:             # normalized 0-1
            bins   = [0, 0.25, 0.50, 0.75, 1.01]
            labels = ["Cold", "Mild", "Warm", "Hot"]
        df["temp_bin"] = pd.cut(t, bins=bins, labels=labels, include_lowest=True)

    df.to_csv(PROC_PATH, index=False)
    print(f"[INFO] EDA-ready data saved -> {PROC_PATH}  shape={df.shape}")
    return df


# =============================================================================
# SECTION 4 - UNIVARIATE ANALYSIS
# =============================================================================
def univariate_analysis(df: pd.DataFrame):
    print("\n" + "="*60)
    print("SECTION 4 - UNIVARIATE ANALYSIS")
    print("="*60)

    # 4a. Histograms + KDE for numeric features
    num_features = [c for c in ["temp", "atemp", "hum", "windspeed", "cnt"]
                    if c in df.columns]
    fig, axes = plt.subplots(1, len(num_features), figsize=(18, 4))
    for ax, col in zip(axes, num_features):
        data = df[col].dropna().astype(float)
        sns.histplot(data, kde=True, ax=ax, color=sns.color_palette(PALETTE)[0])
        ax.set_title(col.capitalize())
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    fig.suptitle("Fig 1 - Distribution of Numeric Features", fontweight="bold")
    plt.tight_layout()
    save_fig("distributions_numeric")

    # 4b. Count plots for categorical features
    cat_features = [c for c in ["season", "weather_label", "workingday", "day_type"]
                    if c in df.columns]
    fig, axes = plt.subplots(1, len(cat_features), figsize=(18, 4))
    for ax, col in zip(axes, cat_features):
        val_order = df[col].dropna().value_counts().index.tolist()
        df_cat = df[[col]].copy()
        df_cat[col] = pd.Categorical(df_cat[col], categories=val_order, ordered=True)
        sns.countplot(data=df_cat, x=col, ax=ax, palette=PALETTE)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Fig 2 - Count of Categorical Features", fontweight="bold")
    plt.tight_layout()
    save_fig("countplots_categorical")

    # 4c. Box plots for numeric features
    fig, axes = plt.subplots(1, len(num_features), figsize=(18, 4))
    for ax, col in zip(axes, num_features):
        data = df[col].dropna().astype(float)
        sns.boxplot(y=data, ax=ax, color=sns.color_palette(PALETTE)[2])
        ax.set_title(col.capitalize())
        ax.set_ylabel(col)
    fig.suptitle("Fig 3 - Box Plots of Numeric Features", fontweight="bold")
    plt.tight_layout()
    save_fig("boxplots_numeric")

    print("  [DONE] Univariate analysis complete.")


# =============================================================================
# SECTION 5 - BIVARIATE ANALYSIS
# =============================================================================
def bivariate_analysis(df: pd.DataFrame):
    print("\n" + "="*60)
    print("SECTION 5 - BIVARIATE ANALYSIS")
    print("="*60)

    cnt_col = "cnt" if "cnt" in df.columns else None
    if cnt_col is None:
        print("[WARN] 'cnt' column not found, skipping bivariate analysis.")
        return

    # 5a. Scatter plots: count vs temp, humidity, windspeed
    scatter_feats = [c for c in ["temp", "hum", "windspeed"] if c in df.columns]
    fig, axes = plt.subplots(1, len(scatter_feats), figsize=(18, 5))
    for ax, feat in zip(axes, scatter_feats):
        sample = df[[feat, cnt_col]].dropna().astype(float).sample(
            min(3000, len(df)), random_state=RANDOM_STATE)
        sns.regplot(data=sample, x=feat, y=cnt_col, ax=ax,
                    scatter_kws={"alpha": 0.3, "s": 15},
                    line_kws={"color": "red"})
        ax.set_title(f"Count vs {feat.capitalize()}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Bike Count")
    fig.suptitle("Fig 4 - Scatter + Regression: Count vs Key Features",
                 fontweight="bold")
    plt.tight_layout()
    save_fig("scatter_regression")

    # 5b. Box + violin: count by season
    if "season" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        s_order = ["Spring", "Summer", "Fall", "Winter"]
        s_order = [o for o in s_order if o in df["season"].dropna().unique()]
        df_s = df[["season", cnt_col]].copy()
        df_s["season"] = pd.Categorical(df_s["season"], categories=s_order, ordered=True)
        df_s = df_s.dropna(subset=["season", cnt_col]).sort_values("season")
        sns.boxplot(data=df_s, x="season", y=cnt_col, palette=PALETTE, ax=axes[0])
        axes[0].set_title("Count by Season (Box)")
        axes[0].set_xlabel("Season"); axes[0].set_ylabel("Bike Count")
        sns.violinplot(data=df_s, x="season", y=cnt_col, palette=PALETTE, ax=axes[1])
        axes[1].set_title("Count by Season (Violin)")
        axes[1].set_xlabel("Season"); axes[1].set_ylabel("Bike Count")
        fig.suptitle("Fig 5 - Count by Season", fontweight="bold")
        plt.tight_layout()
        save_fig("count_by_season")

    # 5c. Box + violin: count by weather
    if "weather_label" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        w_order = ["Clear", "Cloudy", "Light Rain/Snow", "Heavy Rain/Snow"]
        w_order = [o for o in w_order if o in df["weather_label"].dropna().unique()]
        df_w = df[["weather_label", cnt_col]].copy()
        df_w["weather_label"] = pd.Categorical(df_w["weather_label"], categories=w_order, ordered=True)
        df_w = df_w.dropna(subset=["weather_label", cnt_col]).sort_values("weather_label")
        sns.boxplot(data=df_w, x="weather_label", y=cnt_col, palette=PALETTE, ax=axes[0])
        axes[0].set_title("Count by Weather (Box)")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].set_xlabel("Weather"); axes[0].set_ylabel("Bike Count")
        sns.violinplot(data=df_w, x="weather_label", y=cnt_col, palette=PALETTE, ax=axes[1])
        axes[1].set_title("Count by Weather (Violin)")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].set_xlabel("Weather"); axes[1].set_ylabel("Bike Count")
        fig.suptitle("Fig 6 - Count by Weather Condition", fontweight="bold")
        plt.tight_layout()
        save_fig("count_by_weather")

    # 5d. Box: count by hour
    if "hr" in df.columns:
        fig, ax = plt.subplots(figsize=(16, 5))
        df_hr = df[["hr", cnt_col]].dropna().copy()
        df_hr["hr"] = df_hr["hr"].astype(int)
        sns.boxplot(data=df_hr, x="hr", y=cnt_col, palette="husl", ax=ax)
        ax.set_title("Fig 7 - Bike Count Distribution by Hour of Day",
                     fontweight="bold")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Bike Count")
        plt.tight_layout()
        save_fig("count_by_hour_box")

    # 5e. Box + violin: count by day_type
    if "day_type" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        d_order = ["Working Day", "Weekend", "Holiday"]
        d_order = [o for o in d_order if o in df["day_type"].dropna().unique()]
        df_d = df[["day_type", cnt_col]].copy()
        df_d["day_type"] = pd.Categorical(df_d["day_type"], categories=d_order, ordered=True)
        df_d = df_d.dropna(subset=["day_type", cnt_col]).sort_values("day_type")
        sns.boxplot(data=df_d, x="day_type", y=cnt_col, palette=PALETTE, ax=axes[0])
        axes[0].set_title("Count by Day Type (Box)")
        axes[0].set_xlabel("Day Type"); axes[0].set_ylabel("Bike Count")
        sns.violinplot(data=df_d, x="day_type", y=cnt_col, palette=PALETTE, ax=axes[1])
        axes[1].set_title("Count by Day Type (Violin)")
        axes[1].set_xlabel("Day Type"); axes[1].set_ylabel("Bike Count")
        fig.suptitle("Fig 8 - Count by Day Type", fontweight="bold")
        plt.tight_layout()
        save_fig("count_by_daytype")

    # 5f. Bar charts: avg count by month, weekday, weather
    if "mnth" in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        month_avg = df.groupby("mnth")[cnt_col].mean().reset_index()
        month_avg.to_csv(os.path.join(TABLE_DIR, "avg_count_by_month.csv"), index=False)
        sns.barplot(data=month_avg, x="mnth", y=cnt_col, palette="Blues_d",
                    ax=axes[0])
        axes[0].set_title("Avg Count by Month"); axes[0].set_xlabel("Month")
        axes[0].set_ylabel("Avg Bike Count")

        if "weekday_label" in df.columns:
            wd_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            wd_order = [w for w in wd_order if w in df["weekday_label"].dropna().unique()]
            wd_avg = df.groupby("weekday_label")[cnt_col].mean().reindex(wd_order).reset_index()
            wd_avg["weekday_label"] = pd.Categorical(wd_avg["weekday_label"], categories=wd_order, ordered=True)
            wd_avg = wd_avg.sort_values("weekday_label")
            wd_avg.to_csv(os.path.join(TABLE_DIR, "avg_count_by_weekday.csv"), index=False)
            sns.barplot(data=wd_avg, x="weekday_label", y=cnt_col,
                        palette="Greens_d", ax=axes[1])
            axes[1].set_title("Avg Count by Weekday"); axes[1].set_xlabel("Weekday")

        if "weather_label" in df.columns:
            wth_avg = df.groupby("weather_label")[cnt_col].mean().reset_index()
            wth_avg.to_csv(os.path.join(TABLE_DIR, "avg_count_by_weather.csv"), index=False)
            sns.barplot(data=wth_avg, x="weather_label", y=cnt_col,
                        palette="Oranges_d", ax=axes[2])
            axes[2].set_title("Avg Count by Weather")
            axes[2].tick_params(axis="x", rotation=20)

        for ax in axes:
            ax.set_ylabel("Avg Bike Count")
        fig.suptitle("Fig 9 - Average Bike Count by Month / Weekday / Weather",
                     fontweight="bold")
        plt.tight_layout()
        save_fig("avg_count_bar_charts")

    print("  [DONE] Bivariate analysis complete.")


# =============================================================================
# SECTION 6 - MULTIVARIATE AND TIME PATTERNS
# =============================================================================
def multivariate_analysis(df: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("SECTION 6 - MULTIVARIATE AND TIME PATTERNS")
    print("="*60)

    cnt_col = "cnt"
    findings = {}

    # 6a. Hourly demand: working day vs non-working day
    if "hr" in df.columns and "workingday" in df.columns:
        df["workingday"] = pd.to_numeric(df["workingday"], errors="coerce")
        hourly = df.groupby(["hr", "workingday"])[cnt_col].mean().reset_index()
        hourly.to_csv(os.path.join(TABLE_DIR, "avg_count_by_hour_workingday.csv"),
                      index=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        for wd_val, label, color in [(1, "Working Day", "#2196F3"),
                                     (0, "Non-Working Day", "#FF5722")]:
            subset = hourly[hourly["workingday"] == wd_val]
            ax.plot(subset["hr"], subset[cnt_col], marker="o",
                    label=label, color=color, linewidth=2)
        ax.set_title("Fig 10 - Avg Hourly Demand: Working Day vs Non-Working Day",
                     fontweight="bold")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg Bike Count")
        ax.legend(); ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        plt.tight_layout()
        save_fig("hourly_demand_workday_vs_nonworkday")

        wd_peak = int(hourly[hourly["workingday"] == 1].loc[
            hourly[hourly["workingday"] == 1][cnt_col].idxmax(), "hr"])
        findings["peak_hour_workday"] = wd_peak
        print(f"  Peak hour (working day): {wd_peak}:00")

    # 6b. Heatmap: avg demand by hour x weekday
    if "hr" in df.columns and "weekday_int" in df.columns:
        pivot = df.pivot_table(values=cnt_col, index="hr",
                               columns="weekday_int", aggfunc="mean")
        pivot.columns = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][:len(pivot.columns)]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, cmap="YlOrRd", annot=False, fmt=".0f",
                    linewidths=0.5, ax=ax)
        ax.set_title("Fig 11 - Avg Demand Heatmap: Hour x Weekday",
                     fontweight="bold")
        ax.set_xlabel("Weekday"); ax.set_ylabel("Hour of Day")
        plt.tight_layout()
        save_fig("heatmap_hour_weekday")

    # 6c. Heatmap: avg demand by month x hour
    if "mnth" in df.columns and "hr" in df.columns:
        pivot2 = df.pivot_table(values=cnt_col, index="mnth",
                                columns="hr", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(pivot2, cmap="coolwarm", annot=False,
                    linewidths=0.3, ax=ax)
        ax.set_title("Fig 12 - Avg Demand Heatmap: Month x Hour",
                     fontweight="bold")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Month")
        plt.tight_layout()
        save_fig("heatmap_month_hour")

    # 6d. Monthly trend per year (year-over-year)
    if "mnth" in df.columns and "year" in df.columns:
        yoy = df.groupby(["year", "mnth"])[cnt_col].mean().reset_index()
        yoy.to_csv(os.path.join(TABLE_DIR, "avg_count_by_year_month.csv"), index=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        for yr, color in [(2011, "#1565C0"), (2012, "#C62828")]:
            sub = yoy[yoy["year"] == yr]
            ax.plot(sub["mnth"], sub[cnt_col], marker="o",
                    label=str(yr), color=color, linewidth=2)
        ax.set_title("Fig 13 - Year-over-Year Monthly Demand", fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Avg Bike Count")
        ax.legend(title="Year")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.tight_layout()
        save_fig("yoy_monthly_trend")

        avg_2011 = float(df[df["year"] == 2011][cnt_col].mean())
        avg_2012 = float(df[df["year"] == 2012][cnt_col].mean())
        yoy_growth = (avg_2012 - avg_2011) / avg_2011 * 100
        findings["avg_cnt_2011"]   = round(avg_2011, 2)
        findings["avg_cnt_2012"]   = round(avg_2012, 2)
        findings["yoy_growth_pct"] = round(yoy_growth, 2)
        print(f"  YoY growth: {yoy_growth:.1f}%")

    # 6e. Correlation heatmap
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    corr.to_csv(os.path.join(TABLE_DIR, "correlation_matrix.csv"))
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Fig 14 - Annotated Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    save_fig("correlation_heatmap")

    if cnt_col in corr:
        cnt_corr = corr[cnt_col].drop(cnt_col).sort_values(ascending=False)
        findings["top_positive_corr"] = cnt_corr.head(3).to_dict()
        findings["top_negative_corr"] = cnt_corr.tail(3).to_dict()
        print(f"  Top correlations with cnt:\n{cnt_corr.head(5).to_string()}")

    # 6f. Pair plot of key features
    pair_cols = [c for c in ["temp", "hum", "windspeed", cnt_col, "hr"]
                 if c in df.columns]
    pair_df = df[pair_cols].dropna().astype(float).sample(
        min(2000, len(df)), random_state=RANDOM_STATE)
    g = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10},
                     corner=True)
    g.figure.suptitle("Fig 15 - Pair Plot of Key Features", y=1.02,
                       fontweight="bold")
    fpath = os.path.join(FIG_DIR, f"fig{fig_counter[0]+1:02d}_pairplot.png")
    g.figure.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    fig_counter[0] += 1
    print(f"  [SAVED] {fpath}")

    print("  [DONE] Multivariate analysis complete.")
    return findings


# =============================================================================
# SECTION 7 - ANOMALIES AND OUTLIERS
# =============================================================================
def outlier_analysis(df: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("SECTION 7 - ANOMALIES AND OUTLIERS")
    print("="*60)

    outlier_info = {}
    cols = [c for c in ["cnt", "hum", "windspeed"] if c in df.columns]

    for col in cols:
        data = df[col].dropna().astype(float)

        # IQR method
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()

        # Z-score method
        z = np.abs(zscore(data))
        z_outliers = int((z > 3).sum())

        outlier_info[col] = {
            "iqr_outliers":     int(iqr_outliers),
            "zscore_outliers":  z_outliers,
            "min":              float(data.min()),
            "max":              float(data.max()),
        }
        print(f"  {col}: IQR outliers={iqr_outliers}, Z-score outliers={z_outliers}")

    # Box plots for outlier visualisation
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 5))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col].dropna().astype(float), ax=ax,
                    color=sns.color_palette(PALETTE)[4])
        ax.set_title(f"Outliers in {col}")
        ax.set_ylabel(col)
    fig.suptitle("Fig 16 - Outlier Box Plots (IQR Method)", fontweight="bold")
    plt.tight_layout()
    save_fig("outlier_boxplots")

    # Inspect unusual values
    if "hum" in df.columns:
        zero_hum = (df["hum"].astype(float) == 0).sum()
        print(f"  Rows with humidity = 0: {zero_hum}")
        outlier_info["zero_humidity_rows"] = int(zero_hum)

    print("  [DONE] Outlier analysis complete.")
    return outlier_info


# =============================================================================
# SECTION 8 - KEY INSIGHTS
# =============================================================================
def compute_key_insights(df: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("SECTION 8 - KEY INSIGHTS")
    print("="*60)

    cnt_col = "cnt"
    insights = {}

    # Peak hour
    if "hr" in df.columns:
        hr_avg = df.groupby("hr")[cnt_col].mean().dropna()
        if len(hr_avg) > 0:
            peak_hr = int(hr_avg.idxmax())
            insights["peak_hour"] = peak_hr
            insights["peak_hour_avg_count"] = round(float(hr_avg.max()), 1)
            print(f"  1. Peak hour: {peak_hr}:00  (avg={hr_avg.max():.1f} rides)")

    # Busiest season
    if "season" in df.columns:
        season_avg = df.groupby("season")[cnt_col].mean().dropna()
        if len(season_avg) > 0:
            busiest = str(season_avg.idxmax())
            slowest = str(season_avg.idxmin())
            insights["busiest_season"] = busiest
            insights["busiest_season_avg"] = round(float(season_avg.max()), 1)
            insights["slowest_season"] = slowest
            print(f"  2. Busiest season: {busiest} ({season_avg.max():.1f}), "
                  f"Slowest: {slowest} ({season_avg.min():.1f})")

    # Busiest month
    if "mnth" in df.columns:
        mnth_avg = df.groupby("mnth")[cnt_col].mean().dropna()
        if len(mnth_avg) > 0:
            busiest_m = int(mnth_avg.idxmax())
            insights["busiest_month"] = busiest_m
            insights["busiest_month_avg"] = round(float(mnth_avg.max()), 1)
            print(f"  3. Busiest month: {busiest_m} (avg={mnth_avg.max():.1f})")

    # Weather effect
    if "weather_label" in df.columns:
        wth_avg = df.groupby("weather_label")[cnt_col].mean().dropna()
        if len(wth_avg) > 0:
            insights["avg_count_by_weather"] = {k: round(float(v), 1)
                                                for k, v in wth_avg.items()}
            print(f"  4. Weather effect:\n{wth_avg.to_string()}")

    # Morning rush vs evening rush
    if "hr" in df.columns:
        hr_avg_full = df.groupby("hr")[cnt_col].mean()
        top3_hours = hr_avg_full.nlargest(3)
        insights["top3_peak_hours"] = {str(k): round(float(v), 1)
                                        for k, v in top3_hours.items()}
        print(f"  5. Top 3 hours: {top3_hours.index.tolist()}")

    # Temp correlation
    if "temp" in df.columns:
        tc = df[["temp", cnt_col]].dropna().astype(float).corr().loc["temp", cnt_col]
        insights["temp_cnt_correlation"] = round(float(tc), 4)
        print(f"  6. temp <-> cnt correlation: {tc:.4f}")

    # Humidity correlation
    if "hum" in df.columns:
        hc = df[["hum", cnt_col]].dropna().astype(float).corr().loc["hum", cnt_col]
        insights["hum_cnt_correlation"] = round(float(hc), 4)
        print(f"  7. hum  <-> cnt correlation: {hc:.4f}")

    # Day type
    if "day_type" in df.columns:
        dt_avg = df.groupby("day_type")[cnt_col].mean()
        insights["avg_count_by_day_type"] = {k: round(float(v), 1)
                                             for k, v in dt_avg.items()}
        print(f"  8. Day type avg:\n{dt_avg.to_string()}")

    # YoY growth (if year column exists)
    if "year" in df.columns:
        yr_avg = df.groupby("year")[cnt_col].mean()
        if 2011 in yr_avg and 2012 in yr_avg:
            growth = (yr_avg[2012] - yr_avg[2011]) / yr_avg[2011] * 100
            insights["yoy_growth_pct"] = round(float(growth), 2)
            print(f"  9. YoY growth 2011->2012: {growth:.1f}%")

    # Windspeed outliers
    if "windspeed" in df.columns:
        ws = df["windspeed"].astype(float)
        Q1, Q3 = ws.quantile(0.25), ws.quantile(0.75)
        iqr_count = int(((ws < Q1 - 1.5*(Q3-Q1)) | (ws > Q3 + 1.5*(Q3-Q1))).sum())
        insights["windspeed_outliers_iqr"] = iqr_count
        print(f"  10. Windspeed IQR outliers: {iqr_count}")

    return insights


# =============================================================================
# SAVE OUTPUTS
# =============================================================================
def save_outputs(df: pd.DataFrame, info: dict, insights: dict,
                 outlier_info: dict, mv_findings: dict):
    """Persist eda_summary.json and key_findings.md."""
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)

    # describe() tables
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    df[num_cols].describe().to_csv(os.path.join(TABLE_DIR, "describe_numeric.csv"))
    if cat_cols:
        cat_series = [df[c].value_counts().rename(c) for c in cat_cols]
        desc_cat = pd.concat(cat_series, axis=1)
        desc_cat.to_csv(os.path.join(TABLE_DIR, "describe_categorical.csv"))

    # EDA summary JSON
    summary = {
        "dataset":         "Bike Sharing Demand (UCI / OpenML)",
        "rows":            info.get("rows"),
        "columns":         info.get("columns"),
        "missing_values":  info.get("missing_values", {}),
        "duplicates":      info.get("duplicates"),
        "outliers":        outlier_info,
        "multivariate":    mv_findings,
        "key_insights":    insights,
    }

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(i) for i in obj]
        return obj

    summary = _convert(summary)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[SAVED] {SUMMARY_JSON}")

    # Key findings markdown
    ki = insights
    wth = ki.get("avg_count_by_weather", {})
    rain_avg = wth.get("Light Rain/Snow", "N/A")
    clear_avg = wth.get("Clear", "N/A")
    rain_drop = ""
    if isinstance(rain_avg, float) and isinstance(clear_avg, float):
        rain_drop = f"({(clear_avg - rain_avg)/clear_avg*100:.1f}% drop vs Clear)"

    md = f"""# Week 2 - Key EDA Findings: Bike Sharing Demand

## Dataset
- **Source:** UCI / OpenML Bike Sharing Demand (hourly, 2011-2012)
- **Records:** {info.get('rows', 'N/A')}  |  **Features:** {info.get('columns', 'N/A')}

## Top Findings

1. **Peak demand hour is {ki.get('peak_hour', 'N/A')}:00**
   - Average of {ki.get('peak_hour_avg_count', 'N/A')} rides. Demand spikes during morning (8 AM) and evening (5-6 PM) commutes.

2. **Busiest season: {ki.get('busiest_season', 'N/A')}** (avg {ki.get('busiest_season_avg', 'N/A')} rides/hr)
   - {ki.get('slowest_season', 'N/A')} is the slowest season.

3. **Busiest month: Month {ki.get('busiest_month', 'N/A')}** (avg {ki.get('busiest_month_avg', 'N/A')} rides/hr)
   - Mid-year summer/fall months drive the highest demand.

4. **Rain sharply reduces demand** - Light Rain/Snow avg: {rain_avg} {rain_drop}

5. **Temperature is the strongest positive correlate** with count (r = {ki.get('temp_cnt_correlation', 'N/A')})
   - Warmer weather strongly encourages cycling.

6. **Humidity negatively correlates** with count (r = {ki.get('hum_cnt_correlation', 'N/A')})
   - High humidity discourages bike usage.

7. **Top 3 peak hours:** {list(ki.get('top3_peak_hours', {}).keys())}:00
   - Clear bimodal (commute) pattern on working days; unimodal midday peak on weekends.

8. **Year-over-year growth: {ki.get('yoy_growth_pct', 'N/A')}%** (2011 -> 2012)
   - Rapid adoption of the bike-sharing system in its second year.

9. **Average count by day type:** {ki.get('avg_count_by_day_type', {})}
   - Working days show concentrated rush-hour peaks; weekends show spread leisure usage.

10. **Windspeed outliers (IQR):** {ki.get('windspeed_outliers_iqr', 'N/A')} records
    - Extreme windspeed values likely correspond to stormy/unusual weather events.

## Limitations & Next Steps
- Dataset covers only 2011-2012; long-term trends cannot be assessed.
- Casual vs registered users show distinct patterns - separate modelling could improve insights.
- Weather granularity is coarse (4 categories); finer meteorological data could improve demand modelling.
- Next: Feature engineering and predictive modelling (Week 3).
"""
    with open(FINDINGS_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[SAVED] {FINDINGS_MD}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  WEEK 2 - EDA PIPELINE  |  Bike Sharing Demand")
    print("="*60)

    df_raw        = acquire_data()
    info          = initial_analysis(df_raw)
    df            = prepare_data(df_raw)
    univariate_analysis(df)
    bivariate_analysis(df)
    mv_findings   = multivariate_analysis(df)
    outlier_info  = outlier_analysis(df)
    insights      = compute_key_insights(df)
    save_outputs(df, info, insights, outlier_info, mv_findings)

    print("\n" + "="*60)
    print("  ALL DONE - Week 2 EDA pipeline completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()
