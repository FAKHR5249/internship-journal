import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.preprocessor import get_preprocessor

# ==========================================
# Paths
# ==========================================

DATA_PATH = Path("data/processed/featured_taxi_data.parquet")
MODEL_PATH = Path("models/tuned_gradient_boosting.joblib")

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
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Pipeline
# ==========================================

pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(random_state=42))
])

# ==========================================
# Hyperparameters
# ==========================================

param_grid = {
    "model__n_estimators": [50, 100],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5]
}

search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=4,
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=42,
    n_jobs=-1
)

print("Training Tuned Model...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, pred)

print("\nBest Parameters:")
print(search.best_params_)

print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

MODEL_PATH.parent.mkdir(exist_ok=True)

joblib.dump(best_model, MODEL_PATH)

print("\nTuned Model Saved:")
print(MODEL_PATH)