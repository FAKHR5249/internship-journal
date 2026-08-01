import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.models.preprocessor import get_preprocessor

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Random Forest Regression")
print("=" * 60)

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
y = df[target]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Pipeline
# ==========================================================

model = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

pred = model.predict(X_test)

# ==========================================================
# Metrics
# ==========================================================

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

os.makedirs("reports", exist_ok=True)

results.to_csv(
    "reports/random_forest_results.csv",
    index=False
)

print("\nResults")
print(results)

print("\nSaved:")
print("reports/random_forest_results.csv")