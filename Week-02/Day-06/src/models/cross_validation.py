import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.ensemble import GradientBoostingRegressor

from src.models.preprocessor import get_preprocessor

# ======================================================
# Load Dataset
# ======================================================

print("=" * 60)
print("Time Series Cross Validation")
print("=" * 60)

df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

# Sort by pickup time
df = df.sort_values("tpep_pickup_datetime")

# Reduce dataset size to avoid memory error
df = df.head(100000)

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
y = df[target]

# ======================================================
# Pipeline
# ======================================================

pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(random_state=42))
])

# ======================================================
# Time Series Split
# ======================================================

tscv = TimeSeriesSplit(n_splits=5)

scoring = [
    "neg_mean_absolute_error",
    "neg_root_mean_squared_error",
    "r2"
]

scores = cross_validate(
    pipeline,
    X,
    y,
    cv=tscv,
    scoring=scoring,
    n_jobs=1,
    return_train_score=False
)

# ======================================================
# Results
# ======================================================

results = pd.DataFrame({
    "Fold": range(1, 6),
    "MAE": -scores["test_neg_mean_absolute_error"],
    "RMSE": -scores["test_neg_root_mean_squared_error"],
    "R2": scores["test_r2"]
})

print("\nCross Validation Results")
print(results)

print("\nAverage Results")
print(results.mean(numeric_only=True))

# ======================================================
# Save Results
# ======================================================

os.makedirs("reports", exist_ok=True)

results.to_csv(
    "reports/cross_validation_results.csv",
    index=False
)

print("\nSaved:")
print("reports/cross_validation_results.csv")