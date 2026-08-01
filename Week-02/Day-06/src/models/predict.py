import joblib
import pandas as pd
from pathlib import Path

# ==========================================
# Paths
# ==========================================

DATA_PATH = Path("data/processed/featured_taxi_data.parquet")
MODEL_PATH = Path("models/tuned_gradient_boosting.joblib")

# ==========================================
# Load Model
# ==========================================

print("=" * 60)
print("Loading Model")
print("=" * 60)

model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully")

# ==========================================
# Load Dataset
# ==========================================

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

# ==========================================
# Make Predictions
# ==========================================

predictions = model.predict(X)

results = pd.DataFrame({

    "Actual": df[TARGET],

    "Predicted": predictions

})

print("\nFirst 10 Predictions:\n")

print(results.head(10))

# ==========================================
# Save Predictions
# ==========================================

output_path = Path("reports/predictions.csv")

output_path.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(output_path, index=False)

print("\nPredictions Saved Successfully")

print(output_path)