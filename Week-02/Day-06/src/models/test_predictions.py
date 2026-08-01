import pandas as pd
import joblib

print("=" * 60)
print("Generating Test Predictions")
print("=" * 60)

# Load model
model = joblib.load("models/champion_model.joblib")

# Load dataset
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

X = df.drop(
    columns=[c for c in drop_columns if c in df.columns]
)

# Predict
predictions = model.predict(X)

results = pd.DataFrame({
    "Actual": df[target],
    "Predicted": predictions,
    "Absolute Error": abs(df[target] - predictions)
})

results.to_csv(
    "reports/test_predictions.csv",
    index=False
)

print("\nSaved:")
print("reports/test_predictions.csv")

print("\nFirst 10 Predictions:")
print(results.head(10))