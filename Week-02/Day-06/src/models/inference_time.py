import time
import pandas as pd
import joblib

print("=" * 60)
print("Loading Model")
print("=" * 60)

model = joblib.load("models/tuned_gradient_boosting.joblib")

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

# Use first 1000 records
X = X.head(1000)

print("\nMeasuring Inference Time...\n")

start = time.perf_counter()

predictions = model.predict(X)

end = time.perf_counter()

total_time = end - start

per_record = total_time / len(X)

print("=" * 60)
print("Inference Results")
print("=" * 60)

print(f"Records             : {len(X)}")
print(f"Total Time (sec)    : {total_time:.4f}")
print(f"Time per Record     : {per_record:.6f} sec")

result = pd.DataFrame({
    "Records":[len(X)],
    "Total Time (sec)":[total_time],
    "Time per Record (sec)":[per_record]
})

result.to_csv(
    "reports/inference_time.csv",
    index=False
)

print("\nSaved:")
print("reports/inference_time.csv")