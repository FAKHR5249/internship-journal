import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Paths
# ==========================================

DATA_PATH = Path("data/processed/featured_taxi_data.parquet")
MODEL_PATH = Path("models/tuned_gradient_boosting.joblib")

# ==========================================
# Load Dataset
# ==========================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_parquet(DATA_PATH)

TARGET = "trip_duration_minutes"

drop_columns = [

    TARGET,

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

y = df[TARGET]

# ==========================================
# Load Model
# ==========================================

print("\nLoading Tuned Model...")

model = joblib.load(MODEL_PATH)

# ==========================================
# Prediction
# ==========================================

predictions = model.predict(X)

# ==========================================
# Metrics
# ==========================================

mae = mean_absolute_error(y, predictions)

mse = mean_squared_error(y, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y, predictions)

# ==========================================
# Print Results
# ==========================================

print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

print("\nEvaluation Completed Successfully!")