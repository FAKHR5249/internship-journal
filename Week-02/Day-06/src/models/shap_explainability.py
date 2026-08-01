import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("SHAP Explainability")
print("=" * 60)

# ======================================================
# Load Model
# ======================================================

pipeline = joblib.load("models/champion_model.joblib")

# ======================================================
# Load Data
# ======================================================

df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

target = "trip_duration_minutes"

drop_columns = [
    target,
    "tpep_dropoff_datetime",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "tolls_amount",
    "extra",
    "airport_fee",
    "Airport_fee",
    "congestion_surcharge",
    "improvement_surcharge"
]

X = df.drop(columns=[c for c in drop_columns if c in df.columns])

# Small sample for speed
X = X.sample(1000, random_state=42)

# ======================================================
# Transform Features
# ======================================================

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

X_processed = preprocessor.transform(X)

# Convert sparse matrix to dense if needed
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

# Convert to numeric float
X_processed = X_processed.astype(float)

feature_names = preprocessor.get_feature_names_out()
# ======================================================
# SHAP
# ======================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_processed)

os.makedirs("reports", exist_ok=True)

plt.figure(figsize=(12,8))
shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=300)
plt.close()

print("\nSaved:")
print("reports/shap_summary.png")