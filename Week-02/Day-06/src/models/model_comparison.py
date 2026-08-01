import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from src.models.preprocessor import get_preprocessor

# ======================================================
# Load Dataset
# ======================================================

print("=" * 60)
print("Model Comparison")
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ======================================================
# Models
# ======================================================

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Lasso Regression": Lasso(
        alpha=0.1,
        max_iter=5000
    ),

    "Elastic Net": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=5000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=10
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []

# ======================================================
# Train All Models
# ======================================================

for name, estimator in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": round(mae,4),
        "RMSE": round(rmse,4),
        "R2": round(r2,4)
    })

# ======================================================
# Results
# ======================================================

results = pd.DataFrame(results)

results = results.sort_values(
    by="MAE",
    ascending=True
)

os.makedirs("reports", exist_ok=True)

results.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\n")
print("="*60)
print("Final Comparison")
print("="*60)

print(results)

print("\nSaved:")
print("reports/model_comparison.csv")