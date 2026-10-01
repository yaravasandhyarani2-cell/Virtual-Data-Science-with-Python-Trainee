# Week 3: Unsupervised Learning and Clustering Analysis

**Dataset:** [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (200 customers; auto-downloaded from a public GitHub mirror on first run)
**Algorithm:** K-Means (`k-means++`, `n_init=10`, `random_state=42`)
**Author:** Virtual Data Science with Python Internship

## Directory Structure

```text
Week3/
├── data/
│   ├── raw/raw.csv                          # original dataset (cached after download)
│   └── processed/
│       ├── scaled.csv                       # StandardScaler-transformed features
│       ├── k_selection_metrics.csv          # inertia + silhouette per candidate k
│       ├── cluster_profiles.csv             # per-cluster statistics + segment labels
│       └── clustered_customers.csv          # original rows with assigned cluster
├── figures/
│   ├── fig01_elbow_silhouette_curves.png    # elbow curve + silhouette vs k
│   ├── fig02_silhouette_diagrams.png        # silhouette diagram for each k (2-10)
│   ├── fig03_cluster_scatter.png            # segments in original feature space
│   └── fig04_cluster_pca.png                # segments in 2D PCA projection
├── outputs/
│   └── clustering_summary.json              # machine-readable run summary
├── notebooks/
├── src/
│   └── clustering.py                        # full modular pipeline (single entry point)
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python src/clustering.py
```

The script sets its own working directory (same convention as Week 2), downloads and caches the dataset if missing, and writes all outputs relative to `Week3/`.

## Pipeline Steps

1. **Data acquisition** — loads `data/raw/raw.csv` or downloads the Mall Customers CSV from public mirrors (with fallbacks).
2. **Preprocessing** — selects `Annual Income (k$)` and `Spending Score (1-100)`, validates numerics, scales with `StandardScaler` (verified: mean≈0, std≈1).
3. **Optimal k selection** — Elbow Method (inertia/WCSS) + Silhouette Analysis for k = 2..10; the k maximizing the silhouette score is selected automatically.
4. **Final K-Means model** — fit on scaled features; evaluated with Silhouette, Davies-Bouldin (lower = better), and Calinski-Harabasz (higher = better).
5. **Visualizations** — cluster scatter with centroids, PCA projection, elbow curve, and per-k silhouette diagrams, saved to `figures/`.
6. **Cluster profiles** — size, share, income/spending stats, and mean per-cluster silhouette per segment, saved to `data/processed/`.

## Results Summary

**Optimal number of clusters: k = 5** (silhouette = 0.5547; the elbow is also clearly visible at k = 5).

| Metric | Value |
|---|---|
| Silhouette Score | 0.5547 |
| Davies-Bouldin Index | 0.5722 (lower is better) |
| Calinski-Harabasz Index | 248.65 (higher is better) |
| Inertia (WCSS) | 65.57 |

### Cluster Profiles (n = 200)

| Cluster | Size | Avg Income (k$) | Avg Spending Score | Segment |
|---|---|---|---|---|
| 0 | 81 (40.5%) | 55.3 | 49.5 | Standard / Middle Market |
| 1 | 39 (19.5%) | 86.5 | 82.1 | **Target / Premium** — high income, high spending |
| 2 | 22 (11.0%) | 25.7 | 79.4 | Impulsive Spenders |
| 3 | 35 (17.5%) | 88.2 | 17.1 | Careful Wealth |
| 4 | 23 (11.5%) | 26.3 | 20.9 | Budget Conscious |

### Key Takeaways

- **Cluster 1 (high income + high spending)** is the prime marketing target — offer loyalty programs and premium lines.
- **Cluster 3 (high income + low spending)** is the biggest untapped opportunity — targeted re-engagement campaigns could convert their unused purchasing power.
- **Clusters 2 & 4 (low income)** spend at opposite extremes; promotions should be volume/discount-oriented rather than premium.
- **Cluster 0** is the broad middle of the market (40% of customers) with average values on both axes — generic campaigns apply.

## Limitations & Next Steps

- Only two features were clustered; adding `Age` and `Gender` (e.g., one-hot encoded) may reveal finer segments but usually lowers silhouette on this dataset.
- K-Means assumes roughly spherical, equally-sized clusters; DBSCAN or GMM could be compared as alternatives.
- Next: dimensionality reduction (PCA/t-SNE deep dive) and an introduction to hierarchical clustering (Week 4).
