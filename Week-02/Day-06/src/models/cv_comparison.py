import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from src.models.preprocessor import get_preprocessor

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

# Sort for TimeSeriesSplit
df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

drop_columns = [
    "trip_duration_minutes",
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

X = df.drop(columns=drop_columns)
y = df["trip_duration_minutes"]

pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    ))
])

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}

print("\nRunning Random KFold CV...")

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

k_results = cross_validate(
    pipeline,
    X,
    y,
    cv=kfold,
    scoring=scoring,
    n_jobs=-1
)

print("Running TimeSeriesSplit CV...")

tscv = TimeSeriesSplit(n_splits=5)

t_results = cross_validate(
    pipeline,
    X,
    y,
    cv=tscv,
    scoring=scoring,
    n_jobs=-1
)

comparison = pd.DataFrame({
    "Validation": [
        "Random KFold",
        "TimeSeriesSplit"
    ],
    "MAE": [
        -k_results["test_mae"].mean(),
        -t_results["test_mae"].mean()
    ],
    "RMSE": [
        -k_results["test_rmse"].mean(),
        -t_results["test_rmse"].mean()
    ],
    "R2": [
        k_results["test_r2"].mean(),
        t_results["test_r2"].mean()
    ]
})

print("\n")
print("=" * 60)
print("Comparison Results")
print("=" * 60)

print(comparison)

comparison.to_csv(
    "reports/cv_comparison.csv",
    index=False
)

print("\nSaved:")
print("reports/cv_comparison.csv")