import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

from src.models.preprocessor import get_preprocessor

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

# Load Dataset
df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

target = "trip_duration_minutes"

# ==========================================================
# Deployment-Safe Model
# ==========================================================

safe_drop = [
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

X_safe = df.drop(
    columns=[c for c in safe_drop if c in df.columns]
)

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X_safe,
    y,
    test_size=0.20,
    random_state=42
)

safe_model = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(random_state=42))
])

print("\nTraining Deployment-Safe Model...")

safe_model.fit(X_train, y_train)

safe_predictions = safe_model.predict(X_test)

safe_mae = mean_absolute_error(y_test, safe_predictions)
safe_rmse = np.sqrt(mean_squared_error(y_test, safe_predictions))
safe_r2 = r2_score(y_test, safe_predictions)

# ==========================================================
# Oracle Model
# ==========================================================

oracle_drop = [
    target
]

X_oracle = df.drop(
    columns=[c for c in oracle_drop if c in df.columns]
)

X_train, X_test, y_train, y_test = train_test_split(
    X_oracle,
    y,
    test_size=0.20,
    random_state=42
)

oracle_model = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(random_state=42))
])

print("Training Oracle Model...")

oracle_model.fit(X_train, y_train)

oracle_predictions = oracle_model.predict(X_test)

oracle_mae = mean_absolute_error(y_test, oracle_predictions)
oracle_rmse = np.sqrt(mean_squared_error(y_test, oracle_predictions))
oracle_r2 = r2_score(y_test, oracle_predictions)

# ==========================================================
# Results
# ==========================================================

results = pd.DataFrame({
    "Model": [
        "Deployment Safe",
        "Oracle"
    ],
    "MAE": [
        safe_mae,
        oracle_mae
    ],
    "RMSE": [
        safe_rmse,
        oracle_rmse
    ],
    "R2": [
        safe_r2,
        oracle_r2
    ]
})

print("\n" + "=" * 60)
print("Oracle vs Deployment Safe")
print("=" * 60)

print(results)

results.to_csv(
    "reports/oracle_vs_safe.csv",
    index=False
)

print("\nSaved:")
print("reports/oracle_vs_safe.csv")