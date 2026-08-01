import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import GradientBoostingRegressor

from src.models.preprocessor import get_preprocessor

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

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

y_original = df["trip_duration_minutes"]

y_log = np.log1p(y_original)

X_train, X_test, y_train_original, y_test_original = train_test_split(
    X,
    y_original,
    test_size=0.2,
    random_state=42
)

_, _, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)

# ==========================
# Original Target Model
# ==========================

pipeline_original = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    ))
])

pipeline_original.fit(X_train, y_train_original)

pred_original = pipeline_original.predict(X_test)

mae_original = mean_absolute_error(
    y_test_original,
    pred_original
)

rmse_original = np.sqrt(
    mean_squared_error(
        y_test_original,
        pred_original
    )
)

r2_original = r2_score(
    y_test_original,
    pred_original
)

# ==========================
# Log Target Model
# ==========================

pipeline_log = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    ))
])

pipeline_log.fit(X_train, y_train_log)

pred_log = pipeline_log.predict(X_test)

pred_log = np.expm1(pred_log)

mae_log = mean_absolute_error(
    y_test_original,
    pred_log
)

rmse_log = np.sqrt(
    mean_squared_error(
        y_test_original,
        pred_log
    )
)

r2_log = r2_score(
    y_test_original,
    pred_log
)

results = pd.DataFrame({
    "Target": [
        "Original",
        "Log Transformed"
    ],
    "MAE": [
        mae_original,
        mae_log
    ],
    "RMSE": [
        rmse_original,
        rmse_log
    ],
    "R2": [
        r2_original,
        r2_log
    ]
})

print("\n")
print("=" * 60)
print("Comparison")
print("=" * 60)

print(results)

results.to_csv(
    "reports/log_target_comparison.csv",
    index=False
)

print("\nSaved:")
print("reports/log_target_comparison.csv")