import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

from src.models.preprocessor import get_preprocessor

# ======================================================
# Load Dataset
# ======================================================

print("=" * 60)
print("Hyperparameter Tuning")
print("=" * 60)

df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

# Sort by time
df = df.sort_values("tpep_pickup_datetime")

# Reduce dataset size
df = df.head(50000)

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

# ======================================================
# Pipeline
# ======================================================

pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(random_state=42))
])

# ======================================================
# Hyperparameters
# ======================================================

param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
    "model__min_samples_split": [2, 5],
    "model__subsample": [0.8, 1.0]
}

tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=10,
    scoring="neg_root_mean_squared_error",
    cv=tscv,
    random_state=42,
    n_jobs=1,
    verbose=2
)

print("\nTraining...")

search.fit(X, y)

print("\nBest Parameters")
print(search.best_params_)

print("\nBest RMSE")
print(-search.best_score_)

# ======================================================
# Save Results
# ======================================================

os.makedirs("models", exist_ok=True)

import joblib

joblib.dump(search.best_estimator_, "models/best_gradient_boosting.pkl")

print("\nSaved:")
print("models/best_gradient_boosting.pkl")