import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

X = df.drop(columns=[
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
])

y = df["trip_duration_minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Median Baseline": DummyRegressor(strategy="median")
}

results = []

print("\nTraining Baseline Models...\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(name)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}\n")

    results.append([name, mae, rmse, r2])

results = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2"]
)

results.to_csv(
    "reports/baseline_results.csv",
    index=False
)

print("=" * 60)
print(results)

print("\nSaved:")
print("reports/baseline_results.csv")