import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.preprocessor import get_preprocessor

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

# Load Dataset
df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

target = "trip_duration_minutes"

# Remove leakage columns
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

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Lasso Pipeline
model = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", Lasso(alpha=0.1, max_iter=5000))
])

print("\nTraining Lasso Regression...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

results = pd.DataFrame({
    "Model": ["Lasso Regression"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

print("\nResults")
print(results)

results.to_csv(
    "reports/lasso_results.csv",
    index=False
)

print("\nSaved:")
print("reports/lasso_results.csv")