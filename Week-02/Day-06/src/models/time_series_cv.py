import time
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import GradientBoostingRegressor

from src.models.preprocessor import get_preprocessor


# =====================================
# Load Dataset
# =====================================

DATA_PATH = Path("data/processed/featured_taxi_data.parquet")

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_parquet(DATA_PATH)

# Sort by pickup time
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

print(f"Dataset Shape : {df.shape}")

# =====================================
# TimeSeriesSplit
# =====================================

tscv = TimeSeriesSplit(n_splits=5)

pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    ))
])

results = []

print("\nRunning Time Series Cross Validation...\n")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

    print(f"Fold {fold}")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    start = time.time()

    pipeline.fit(X_train, y_train)

    train_time = time.time() - start

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print()

    results.append([
        fold,
        mae,
        rmse,
        r2,
        train_time
    ])

results = pd.DataFrame(
    results,
    columns=[
        "Fold",
        "MAE",
        "RMSE",
        "R2",
        "Training Time"
    ]
)

print("=" * 60)
print("Cross Validation Results")
print("=" * 60)

print(results)

print("\nAverage Results")

print(results.mean(numeric_only=True))

results.to_csv(
    "reports/time_series_cv.csv",
    index=False
)

print("\nResults Saved")

print("reports/time_series_cv.csv")