# Day 2 — Dashboard Development: Setup Guide

## Files
- `app.py` — main Streamlit dashboard
- `requirements.txt` — dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Before running — edit these paths at the top of `app.py`

```python
DATA_PATH = "segmented_customers.csv"
CLUSTER_PROFILE_PATH = "cluster_profile.csv"
CLUSTER_NAMES_PATH = "business_cluster_names.csv"
KMEANS_EVAL_PATH = "kmeans_evaluation.csv"
MODEL_PATH = "models/kmeans_model.pkl"
SCALER_PATH = "models/standard_scaler.pkl"
PIPELINE_PATH = "pipeline/customer_segmentation_pipeline.pkl"
```

Point these at the real locations of your files (e.g. copy `app.py` into your
`Day 11/Customer Segmentation Model Development/` folder so relative paths just work,
or use full paths).

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Required-features checklist (Day 2 spec)

| Requirement | Where it is |
|---|---|
| Navigation/sidebar | `st.sidebar.radio` — 7 pages |
| Filters | Sidebar: segment multiselect, age slider, income slider |
| Interactive charts | Plotly pie, bar, box, histogram, scatter |
| KPI cards | `st.metric` on Overview, Segment Explorer, Model Info |
| Tables | `st.dataframe` on every page |
| Customer search | Dedicated "🔍 Customer Search" page |
| Segment selection | Dropdowns on Segment Explorer, Business Recommendations |
| Download option | `st.download_button` on Overview, Segment Explorer, Search |

## Known placeholders to fill in
- **Business Recommendations page**: currently a placeholder — paste your actual
  Module 8 text (from `Executive_Summary.pdf` / the notebook) per segment.
- **Predict New Customer page**: assumes `pipeline.predict()` or
  `model.predict(scaler.transform(input))` — adjust if your pipeline's `.predict()`
  signature differs.
- Column names assumed: `Customer_Age`, `Income`, `Total_Spending`, `Recency`,
  `Customer_Tenure`, `Family_Size`, `Total_Children`, `NumWebPurchases`,
  `NumStorePurchases`, `NumCatalogPurchases`, `Purchase_Frequency`,
  `Accepted_Campaigns`, `Cluster`. If your actual CSV uses different names,
  update the `FEATURES` list near the top of `app.py`.
