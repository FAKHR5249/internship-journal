import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

from src.models.preprocessor import get_preprocessor

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    ))
])

print("\nTraining Model...")

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

results = []

segments = {
    "All Trips": np.ones(len(X_test), dtype=bool),
    "Peak Hours": X_test["is_peak_hour"] == 1,
    "Off Peak": X_test["is_peak_hour"] == 0,
    "Weekend": X_test["is_weekend"] == 1,
    "Airport Pickup": X_test["airport_pickup"] == 1,
    "Airport Dropoff": X_test["airport_dropoff"] == 1,
    "Long Trips (>30 min)": y_test > 30
}

for name, mask in segments.items():

    if mask.sum() == 0:
        continue

    y_true = y_test[mask]
    y_pred = pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    median_error = np.median(np.abs(y_true - y_pred))
    p90 = np.percentile(np.abs(y_true - y_pred), 90)

    within5 = (
        np.mean(np.abs(y_true - y_pred) <= 5) * 100
    )

    results.append([
        name,
        len(y_true),
        mae,
        rmse,
        median_error,
        p90,
        within5
    ])

results = pd.DataFrame(
    results,
    columns=[
        "Segment",
        "Trips",
        "MAE",
        "RMSE",
        "Median Error",
        "P90 Error",
        "Within 5 Minutes (%)"
    ]
)

print("\n")
print("=" * 60)
print("Segment Performance")
print("=" * 60)

print(results)

results.to_csv(
    "reports/segment_performance.csv",
    index=False
)

print("\nSaved:")
print("reports/segment_performance.csv")