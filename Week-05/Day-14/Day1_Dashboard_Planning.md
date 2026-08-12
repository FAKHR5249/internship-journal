# DAY 1 — Dashboard Planning & Application Setup
**Objective:** Design the structure of the final ML application before development.

---

## 1. Review of Previous Project Outputs

Project traced across two phases:

| Phase | Location | Contents |
|---|---|---|
| Module 6 — Model Development | `Day 11/Customer Segmentation Model Development/` | Processed data, trained models, pipelines, plots, 30+ evaluation reports |
| Module 8 — Business Insights | `files (1)/` | Executive summary, full business report, segmented customer data, analysis notebook |

---

## 2. Final Dataset

- **File:** `data/processed/clustering_dataset.csv`
- Two scaled versions also available for reference/reproducibility:
  - `standard_scaled_dataset.csv`
  - `minmax_scaled_dataset.csv`
- **Final labeled output (post-clustering):** `segmented_customers.csv` (Module 8 folder) — this is the dataset the dashboard will actually read from, since it already contains cluster assignments.

---

## 3. Final Engineered Features

Identified from the `plots/` boxplot set and `cluster_profile.csv` columns:

- Customer_Age
- Income
- Total_Spending
- Recency
- Customer_Tenure
- Family_Size
- Total_Children
- NumWebPurchases
- NumStorePurchases
- NumCatalogPurchases
- Purchase_Frequency
- Accepted_Campaigns

Supporting reports: `feature_distribution_report.csv`, `high_correlation_report.csv`, `pca_variance.csv`, `outlier_report.csv`, `eda_summary_statistics.csv`

---

## 4. Final Customer Segmentation Model

- **Selected Algorithm:** **K-Means**
- **Decision basis:** `final_model_selection.csv`, `algorithm_comparison.csv`, `kmeans_vs_hierarchical.csv`
- **Supporting evaluation:** `kmeans_evaluation.csv`, `silhouette_scores.csv`, `davies_bouldin_scores.csv`, `elbow_results.csv`, `optimal_k.txt`, `stability_analysis.csv`, `bootstrap_stability.csv`
- GMM and Hierarchical were built as comparison baselines (`gmm_evaluation.csv`, `hierarchical_evaluation.csv`, `dbscan_evaluation.csv`) but were not selected as final.

---

## 5. Saved Model Files (to be loaded in the app)

| File | Purpose |
|---|---|
| `models/kmeans_model.pkl` | Final trained clustering model |
| `models/standard_scaler.pkl` | Feature scaler used before prediction (matches KMeans training) |
| `pipeline/preprocessing_pipeline.pkl` | Full preprocessing pipeline (cleaning + scaling) |
| `pipeline/customer_segmentation_pipeline.pkl` | End-to-end pipeline: raw input → cluster label |

> Note: `gmm_model.pkl` and `minmax_scaler.pkl` are not needed in the final app since KMeans (with standard scaling) is the selected model — keep only for reference/audit trail.

---

## 6. Customer Segment Labels

- **Numeric cluster IDs → business names mapping:** `business_cluster_names.csv`
- **Per-customer assignment:** `customer_clusters.csv` / `segmented_customers.csv`
- **Segment size & profile:** `cluster_size.csv`, `cluster_profile.csv`, `cluster_centers.csv`

Example from `cluster_profile.csv`:
| Cluster | Avg Age | Avg Income | Avg Spending | Avg Recency |
|---|---|---|---|---|
| 0 | 55.5 | 37,699 | 151.5 | 49.2 |
| 1 | 59.2 | 70,103 | 1,164.1 | 49.0 |

(Cluster 0 = lower income/spend segment, Cluster 1 = higher income/spend segment — final descriptive names come from `business_cluster_names.csv`.)

---

## 7. Business Recommendations (Module 8)

Sourced from:
- `Executive_Summary.pdf`
- `Full_Business_Report.pdf`
- `Module8_Business_Insights.ipynb`

These will populate a **"Business Recommendations"** section per segment on the dashboard (marketing strategy, targeting suggestions, campaign focus per cluster).

---

## 8. Information to Show on the Dashboard

| Section | Content |
|---|---|
| **Overview / KPIs** | Total customers, number of segments, avg income/spending overall |
| **Segment Explorer** | Filter by cluster → see size, avg age, income, spending, tenure, recency |
| **Segment Profiles** | Radar/bar chart per cluster using `cluster_profile.csv` values |
| **Feature Distributions** | Boxplots per feature per segment (reuse existing `plots/` visuals or regenerate dynamically) |
| **Business Recommendations** | Text panel per segment, pulled from Module 8 report |
| **Model Info** | Which model (KMeans), number of clusters, silhouette score, last updated date |
| **Live Prediction Tool** | Upload/enter new customer data → app runs `customer_segmentation_pipeline.pkl` → shows predicted segment + recommendation |

---

## 9. Dashboard Navigation Structure (Streamlit sidebar)

```
📊 Customer Segmentation Dashboard
├── 🏠 Overview               (KPIs, total customers, segment counts)
├── 👥 Segment Explorer        (filter/select cluster, view profile table & charts)
├── 📈 Feature Analysis        (boxplots / distributions per feature per segment)
├── 💡 Business Recommendations (per-segment strategy from Module 8)
├── 🧠 Model Info               (algorithm, metrics, evaluation comparison)
└── 🔮 Predict New Customer     (input form → live cluster prediction)
```

---

## 10. Application Technology Selection

- **Chosen:** **Streamlit**
- **Justification:**
  - Loads `.pkl` model/pipeline files directly with Python — no separate backend needed
  - Fast to build multi-page apps with sidebar navigation (matches structure above)
  - Native support for pandas DataFrames, matplotlib/plotly charts — fits existing `plots/` and `.csv` reports
  - Easy deployment (Streamlit Community Cloud / internal server) for an intern-level project timeline

---

## Next Step (Day 2 preview)
Build the Streamlit app skeleton: `app.py` with `st.sidebar` navigation calling separate page functions, loading `kmeans_model.pkl` + `preprocessing_pipeline.pkl` + `segmented_customers.csv`.
