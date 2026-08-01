import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
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
y = df["trip_duration_minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

sizes = [0.10, 0.25, 0.50, 1.00]

results = []

print("\nRunning Learning Curve Experiment...\n")

for size in sizes:

    X_small = X_train.sample(
        frac=size,
        random_state=42
    )

    y_small = y_train.loc[X_small.index]

    pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", GradientBoostingRegressor(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        ))
    ])

    start = time.time()

    pipeline.fit(X_small, y_small)

    training_time = time.time() - start

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    print(f"{int(size*100)}% Data")
    print(f"MAE  : {mae:.4f}")
    print(f"Time : {training_time:.2f} sec\n")

    results.append([
        int(size*100),
        mae,
        training_time
    ])

results = pd.DataFrame(
    results,
    columns=[
        "Training %",
        "MAE",
        "Training Time"
    ]
)

print("=" * 60)
print(results)

results.to_csv(
    "reports/learning_curve.csv",
    index=False
)

plt.figure(figsize=(8,5))

plt.plot(
    results["Training %"],
    results["MAE"],
    marker="o"
)

plt.title("Learning Curve")

plt.xlabel("Training Data (%)")

plt.ylabel("MAE")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reports/figures/learning_curve.png",
    dpi=300
)

plt.show()

print("\nSaved:")
print("reports/learning_curve.csv")
print("reports/figures/learning_curve.png")