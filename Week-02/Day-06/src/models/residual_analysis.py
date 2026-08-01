import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from src.models.preprocessor import get_preprocessor

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

os.makedirs("reports/figures", exist_ok=True)

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

X = df.drop(columns=[c for c in drop_columns if c in df.columns])
y = df["trip_duration_minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(random_state=42))
])

print("\nTraining Model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
residuals = y_test - predictions

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Residual": residuals
})

results.to_csv("reports/residual_analysis.csv", index=False)

# Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions, alpha=0.3)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("reports/figures/predicted_vs_actual.png")
plt.close()

# Residual Distribution
plt.figure(figsize=(7,5))
plt.hist(residuals, bins=50)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reports/figures/residual_distribution.png")
plt.close()

# Residual vs Prediction
plt.figure(figsize=(7,5))
plt.scatter(predictions, residuals, alpha=0.3)
plt.axhline(0, linestyle="--")
plt.xlabel("Prediction")
plt.ylabel("Residual")
plt.title("Residual vs Prediction")
plt.tight_layout()
plt.savefig("reports/figures/residual_vs_prediction.png")
plt.close()

print("\nSaved:")
print("reports/residual_analysis.csv")
print("reports/figures/predicted_vs_actual.png")
print("reports/figures/residual_distribution.png")
print("reports/figures/residual_vs_prediction.png")