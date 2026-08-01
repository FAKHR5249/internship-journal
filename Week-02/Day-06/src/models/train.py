import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyRegressor
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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.models.preprocessor import get_preprocessor


# ==========================================
# Paths
# ==========================================

DATA_PATH = Path("data/processed/featured_taxi_data.parquet")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


# ==========================================
# Load Dataset
# ==========================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

df = pd.read_parquet(DATA_PATH)

print(df.shape)
print()


# ==========================================
# Target
# ==========================================

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


# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("Train:", X_train.shape)
print("Test :", X_test.shape)


# ==========================================
# Models
# ==========================================

models = {

    "Dummy":
        DummyRegressor(strategy="median"),

    "Linear Regression":
        LinearRegression(),

    "Ridge":
        Ridge(alpha=1.0),

    "Lasso":
        Lasso(alpha=0.01),

    "ElasticNet":
        ElasticNet(
            alpha=0.01,
            l1_ratio=0.5
        ),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=15
        ),

    "Random Forest":
RandomForestRegressor(
    n_estimators=10,
    max_depth=10,
    random_state=42,
    n_jobs=-1
),
    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=50,
            random_state=42
        )

}


results = []

best_model = None

best_mae = float("inf")

best_name = ""
# ==========================================
# Training Loop
# ==========================================

print("\n" + "=" * 60)
print("Training Started")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}")

    pipeline = Pipeline(

        steps=[

            ("preprocessor", get_preprocessor()),

            ("model", model)

        ]

    )

    start = time.time()

    pipeline.fit(

        X_train,

        y_train

    )

    training_time = time.time() - start

    predictions = pipeline.predict(

        X_test

    )

    mae = mean_absolute_error(

        y_test,

        predictions

    )

    mse = mean_squared_error(

        y_test,

        predictions

    )

    rmse = np.sqrt(

        mse

    )

    r2 = r2_score(

        y_test,

        predictions

    )

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print(f"Time : {training_time:.2f} sec")

    results.append({

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "Training Time": training_time

    })

    if mae < best_mae:

        best_mae = mae

        best_model = pipeline

        best_name = name
        # ==========================================
# Save Champion Model
# ==========================================

print("\n" + "=" * 60)
print("Saving Champion Model")
print("=" * 60)

MODEL_PATH = MODEL_DIR / "champion_model.joblib"

joblib.dump(

    best_model,

    MODEL_PATH

)

print(f"Champion Model : {best_name}")
print(f"Best MAE       : {best_mae:.4f}")
print(f"Saved To       : {MODEL_PATH}")


# ==========================================
# Results DataFrame
# ==========================================

results_df = pd.DataFrame(

    results

)

results_df = results_df.sort_values(

    by="MAE"

)

CSV_PATH = REPORT_DIR / "model_comparison.csv"

results_df.to_csv(

    CSV_PATH,

    index=False

)


# ==========================================
# Print Results
# ==========================================

print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)

print(results_df)

print("\nCSV Saved :", CSV_PATH)


# ==========================================
# Best Model
# ==========================================

print("\n" + "=" * 60)
print("Best Model Summary")
print("=" * 60)

print(f"Best Model : {best_name}")
print(f"MAE        : {best_mae:.4f}")

print("\nTraining Completed Successfully!")