"""
Week 3: Unsupervised Learning and Clustering Analysis
Dataset: Mall Customer Segmentation (Kaggle-style public dataset)
Author: Virtual Data Science with Python Internship

Pipeline:
  1. Data acquisition  -> data/raw/raw.csv
  2. Preprocessing (StandardScaler) -> data/processed/scaled.csv
  3. Optimal k selection (Elbow Method + Silhouette Analysis)
  4. K-Means clustering + evaluation (Silhouette / Davies-Bouldin / Calinski-Harabasz)
  5. Visualizations -> figures/
  6. Cluster profile summaries -> data/processed/
"""

# ---------------------------------------------
# 0. SETUP - ensure working directory is Week3
# ---------------------------------------------
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Locate the Week3 root dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))   # .../Week3/src
week3_root = os.path.dirname(script_dir)                  # .../Week3
os.chdir(week3_root)
print(f"[INFO] Working directory set to: {os.getcwd()}")

# ---------------------------------------------
# 1. IMPORTS
# ---------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA

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
RAW_PATH   = os.path.join("data", "raw", "raw.csv")
FIG_DIR    = os.path.join("figures")
PROC_DIR   = os.path.join("data", "processed")
SUMMARY_JSON = os.path.join("outputs", "clustering_summary.json")

K_RANGE = range(2, 11)   # candidate cluster counts
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]

os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)

fig_counter = [0]

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
    """Load Mall Customer Segmentation; download from a public mirror if absent."""
    print("\n" + "="*60)
    print("SECTION 1 - DATA ACQUISITION")
    print("="*60)

    if os.path.exists(RAW_PATH):
        print(f"[INFO] Raw data already exists at {RAW_PATH}. Loading ...")
        df = pd.read_csv(RAW_PATH)
        print(f"[INFO] Shape: {df.shape}")
        return df

    print("[INFO] Raw data not found. Downloading Mall Customers dataset ...")
    import urllib.request
    import io

    # Public mirrors of the Mall Customer Segmentation dataset, tried in order
    mirror_urls = [
        ("https://gist.githubusercontent.com/pravalliyaram/"
         "5c05f43d2351249927b8a3f3cc3e5ecf/raw/Mall_Customers.csv",
         "Public GitHub Gist mirror (pravalliyaram)"),
        ("https://raw.githubusercontent.com/tirthajyoti/"
         "Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv",
         "Public GitHub mirror (tirthajyoti/Machine-Learning-with-Python)"),
        ("https://raw.githubusercontent.com/kennedykwangari/"
         "Mall-Customer-Segmentation-Data/master/Mall_Customers.csv",
         "Public GitHub mirror (kennedykwangari/Mall-Customer-Segmentation-Data)"),
    ]

    df, source = None, None
    errors = []
    for url, label in mirror_urls:
        try:
            print(f"[INFO] Trying {label} ...")
            with urllib.request.urlopen(url, timeout=60) as resp:
                df = pd.read_csv(io.BytesIO(resp.read()))
            source = label
            break
        except Exception as e:
            errors.append(f"{label}: {e}")
            df = None

    if df is None:
        raise RuntimeError(
            "[ERROR] All download mirrors failed:\n  - "
            + "\n  - ".join(errors)
            + f"\nPlace Mall_Customers.csv manually at {RAW_PATH} and re-run."
        )

    print(f"[INFO] Data source: {source}")
    df.to_csv(RAW_PATH, index=False)
    print(f"[INFO] Raw data saved -> {RAW_PATH}")
    print(f"[INFO] Shape: {df.shape}")
    return df


# =============================================================================
# SECTION 2 - PREPROCESSING (StandardScaler)
# =============================================================================
def preprocess_data(df: pd.DataFrame):
    """Select numeric features, validate, and scale with StandardScaler."""
    print("\n" + "="*60)
    print("SECTION 2 - PREPROCESSING (StandardScaler)")
    print("="*60)

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"[ERROR] Expected feature columns not found: {missing}. "
                       f"Available columns: {df.columns.tolist()}")

    X_raw = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    n_missing = int(X_raw.isnull().sum().sum())
    n_dupes   = int(X_raw.duplicated().sum())
    print(f"  Rows: {len(X_raw)} | Missing cells: {n_missing} | "
          f"Duplicate feature rows: {n_dupes}")
    X_raw = X_raw.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"  Scaler: mean={scaler.mean_.round(3).tolist()}, "
          f"scale={scaler.scale_.round(3).tolist()}")
    print(f"  Scaled check -> mean~0: {np.allclose(X_scaled.mean(axis=0), 0)}, "
          f"std~1: {np.allclose(X_scaled.std(axis=0), 1)}")

    # Persist scaled features aligned with original identifiers
    id_cols = [c for c in ["CustomerID", "Gender", "Age"] if c in df.columns]
    scaled_df = df.loc[X_raw.index, id_cols + FEATURES].copy()
    for i, col in enumerate(FEATURES):
        scaled_df[f"scaled_{col}"] = X_scaled[:, i]
    scaled_path = os.path.join(PROC_DIR, "scaled.csv")
    scaled_df.to_csv(scaled_path, index=False)
    print(f"  [SAVED] {scaled_path}")

    return X_raw, X_scaled, scaler


# =============================================================================
# SECTION 3 - OPTIMAL k: ELBOW METHOD + SILHOUETTE ANALYSIS
# =============================================================================
def select_optimal_k(X_scaled: np.ndarray) -> int:
    """Compute inertia and silhouette over K_RANGE; pick k with max silhouette."""
    print("\n" + "="*60)
    print("SECTION 3 - OPTIMAL k (Elbow + Silhouette)")
    print("="*60)

    ks = list(K_RANGE)
    inertias, silhouettes = [], []

    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"  k={k:2d} | inertia={km.inertia_:10.2f} | silhouette={sil:.4f}")

    metrics = pd.DataFrame({"k": ks, "inertia": inertias,
                            "silhouette": silhouettes})
    metrics.to_csv(os.path.join(PROC_DIR, "k_selection_metrics.csv"), index=False)

    best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    print(f"  [INFO] Best k by silhouette score: {best_k} "
          f"(score={max(silhouettes):.4f})")

    # -- Figure: elbow curve + silhouette curve side by side ------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ks, inertias, marker="o", color="#1565C0", linewidth=2)
    axes[0].axvline(best_k, color="#C62828", linestyle="--",
                    label=f"selected k={best_k}")
    axes[0].set_title("Elbow Method (Inertia vs k)", fontweight="bold")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].xaxis.set_major_locator(mticker.MultipleLocator(1))
    axes[0].legend()

    axes[1].plot(ks, silhouettes, marker="s", color="#2E7D32", linewidth=2)
    axes[1].axvline(best_k, color="#C62828", linestyle="--",
                    label=f"best silhouette k={best_k}")
    axes[1].set_title("Silhouette Score vs k", fontweight="bold")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Mean Silhouette Score")
    axes[1].xaxis.set_major_locator(mticker.MultipleLocator(1))
    axes[1].legend()
    fig.suptitle("Fig - Optimal Cluster Count Selection", fontweight="bold")
    plt.tight_layout()
    save_fig("elbow_silhouette_curves")

    # -- Figure: silhouette diagram for each k --------------------------------
    n_cols = 3
    n_rows = int(np.ceil(len(ks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.get_cmap(PALETTE)
    for ax, k in zip(axes, ks):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil_avg = silhouette_score(X_scaled, labels)
        sil_vals = silhouette_samples(X_scaled, labels)

        y_lower = 10
        for i in range(k):
            c_vals = np.sort(sil_vals[labels == i])
            c_size = len(c_vals)
            y_upper = y_lower + c_size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_vals,
                             facecolor=cmap(i % 10), edgecolor="none", alpha=0.85)
            ax.text(-0.06, y_lower + 0.5 * c_size, str(i), fontsize=8,
                    ha="right")
            y_lower = y_upper + 10

        ax.axvline(sil_avg, color="red", linestyle="--")
        ax.set_title(f"k={k} (avg silhouette={sil_avg:.3f})")
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("Cluster label")
        ax.set_yticks([])
        ax.set_xlim([-0.1, 1.0])
    for ax in axes[len(ks):]:
        ax.axis("off")
    fig.suptitle("Fig - Silhouette Analysis per k", fontweight="bold")
    plt.tight_layout()
    save_fig("silhouette_diagrams")

    return best_k


# =============================================================================
# SECTION 4 - FINAL K-MEANS MODEL + EVALUATION
# =============================================================================
def fit_final_model(X_scaled: np.ndarray, best_k: int):
    """Fit final K-Means model and compute evaluation metrics."""
    print("\n" + "="*60)
    print("SECTION 4 - FINAL K-MEANS MODEL")
    print("="*60)

    km = KMeans(n_clusters=best_k, init="k-means++", n_init=10,
                random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)

    metrics = {
        "best_k":              best_k,
        "silhouette":          round(float(silhouette_score(X_scaled, labels)), 4),
        "davies_bouldin":      round(float(davies_bouldin_score(X_scaled, labels)), 4),
        "calinski_harabasz":   round(float(calinski_harabasz_score(X_scaled, labels)), 2),
        "inertia":             round(float(km.inertia_), 2),
        "cluster_sizes":       {int(i): int(n) for i, n in
                                zip(*np.unique(labels, return_counts=True))},
    }
    print(f"  Silhouette:        {metrics['silhouette']}")
    print(f"  Davies-Bouldin:    {metrics['davies_bouldin']}  (lower is better)")
    print(f"  Calinski-Harabasz: {metrics['calinski_harabasz']}  (higher is better)")
    print(f"  Inertia:           {metrics['inertia']}")
    print(f"  Cluster sizes:     {metrics['cluster_sizes']}")

    return km, labels, metrics


# =============================================================================
# SECTION 5 - VISUALIZATIONS
# =============================================================================
def visualize_clusters(X_raw: pd.DataFrame, X_scaled: np.ndarray,
                       labels: np.ndarray, centers: np.ndarray):
    """Scatter plots of clusters in feature space and PCA space."""
    print("\n" + "="*60)
    print("SECTION 5 - VISUALIZATIONS")
    print("="*60)

    x_col, y_col = FEATURES
    palette = sns.color_palette(PALETTE, n_colors=len(np.unique(labels)))

    # -- Figure: cluster scatter in original feature space ---------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, color in enumerate(palette):
        pts = X_raw[labels == i]
        ax.scatter(pts[x_col], pts[y_col], s=45, alpha=0.75, color=color,
                   edgecolor="white", linewidth=0.5, label=f"Cluster {i}")
    # Centers back-transformed to original units
    c_mean = X_raw.mean().values
    c_std = X_raw.std().values
    centers_orig = centers * c_std + c_mean
    ax.scatter(centers_orig[:, 0], centers_orig[:, 1], s=280, marker="X",
               c="black", edgecolor="white", linewidth=1.5, label="Centroids",
               zorder=5)
    ax.set_title("Fig - K-Means Customer Segments (Scaled Features)",
                 fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(title="Segment", loc="best")
    plt.tight_layout()
    save_fig("cluster_scatter")

    # -- Figure: PCA 2D projection colored by cluster --------------------------
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, color in enumerate(palette):
        ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], s=45,
                   alpha=0.75, color=color, edgecolor="white", linewidth=0.5,
                   label=f"Cluster {i}")
    centers_pca = pca.transform(centers)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=280, marker="X",
               c="black", edgecolor="white", linewidth=1.5, label="Centroids",
               zorder=5)
    ax.set_title("Fig - Cluster Separation in PCA Space",
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({evr[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1f}% variance)")
    ax.legend(title="Segment", loc="best")
    plt.tight_layout()
    save_fig("cluster_pca")

    print("  [DONE] Visualizations complete.")


# =============================================================================
# SECTION 6 - CLUSTER PROFILE SUMMARIES
# =============================================================================
def build_profiles(X_raw: pd.DataFrame, labels: np.ndarray,
                   sil_vals: np.ndarray) -> pd.DataFrame:
    """Aggregate per-cluster statistics and save profile summaries."""
    print("\n" + "="*60)
    print("SECTION 6 - CLUSTER PROFILES")
    print("="*60)

    dfp = X_raw.copy()
    dfp["cluster"] = labels

    # Align per-sample silhouette values with dfp's index (robust to any drops)
    sil_series = pd.Series(sil_vals, index=dfp.index)

    profile = dfp.groupby("cluster").agg(
        size=("cluster", "size"),
        pct=("cluster", lambda s: round(100 * len(s) / len(dfp), 1)),
        income_mean=(FEATURES[0], "mean"),
        income_min=(FEATURES[0], "min"),
        income_max=(FEATURES[0], "max"),
        score_mean=(FEATURES[1], "mean"),
        score_min=(FEATURES[1], "min"),
        score_max=(FEATURES[1], "max"),
        silhouette_mean=("cluster", lambda s: float(sil_series.loc[s.index].mean())),
    ).round(2)
    profile.index.name = "cluster"
    profile.to_csv(os.path.join(PROC_DIR, "cluster_profiles.csv"))

    print("\n" + profile.to_string())

    # Segment interpretation (income x spending quadrant labels)
    inc_med = dfp[FEATURES[0]].median()
    sco_med = dfp[FEATURES[1]].median()

    def interpret(row) -> str:
        inc, sco = row["income_mean"], row["score_mean"]
        near = lambda a, b, tol=8: abs(a - b) <= tol
        if near(inc, inc_med) and near(sco, sco_med):
            return "Standard / Middle Market (near-median income and spending)"
        if inc >= inc_med and sco >= sco_med:
            return "Target / Premium (high income, high spending)"
        if inc >= inc_med and sco < sco_med:
            return "Careful Wealth (high income, low spending)"
        if inc < inc_med and sco >= sco_med:
            return "Impulsive Spenders (low income, high spending)"
        return "Budget Conscious (low income, low spending)"

    profile["segment"] = profile.apply(interpret, axis=1)
    profile.to_csv(os.path.join(PROC_DIR, "cluster_profiles.csv"))

    # Full labelled dataset for downstream use
    labelled = X_raw.copy()
    labelled["cluster"] = labels
    labelled.to_csv(os.path.join(PROC_DIR, "clustered_customers.csv"), index=False)

    print(f"\n  [SAVED] {os.path.join(PROC_DIR, 'cluster_profiles.csv')}")
    print(f"  [SAVED] {os.path.join(PROC_DIR, 'clustered_customers.csv')}")
    print("  [DONE] Cluster profiling complete.")
    return profile


# =============================================================================
# SAVE OUTPUTS
# =============================================================================
def save_summary(metrics: dict, profile: pd.DataFrame,
                 raw_shape: tuple, features: list):
    """Persist clustering_summary.json."""
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)

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

    summary = {
        "dataset":    "Mall Customer Segmentation",
        "rows":       int(raw_shape[0]),
        "columns":    int(raw_shape[1]),
        "features_used": features,
        "scaling":    "StandardScaler",
        "algorithm":  "KMeans (k-means++, n_init=10, random_state=42)",
        "metrics":    metrics,
        "profiles":   json.loads(profile.reset_index().to_json(orient="records")),
    }
    summary = _convert(summary)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[SAVED] {SUMMARY_JSON}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  WEEK 3 - CLUSTERING PIPELINE  |  Mall Customer Segmentation")
    print("="*60)

    # 1. Acquire data
    df = acquire_data()

    # 2. Preprocess + scale
    X_raw, X_scaled, scaler = preprocess_data(df)

    # 3. Optimal k (elbow + silhouette)
    best_k = select_optimal_k(X_scaled)

    # 4. Final K-Means model + evaluation metrics
    km, labels, metrics = fit_final_model(X_scaled, best_k)

    # Per-sample silhouette values for profiling
    sil_vals = silhouette_samples(X_scaled, labels)

    # 5. Visualizations
    visualize_clusters(X_raw, X_scaled, labels, km.cluster_centers_)

    # 6. Cluster profiles
    profile = build_profiles(X_raw, labels, sil_vals)

    # Save summary outputs
    save_summary(metrics, profile, df.shape, list(X_raw.columns))

    print("\n" + "="*60)
    print("  ALL DONE - Week 3 clustering pipeline completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()
