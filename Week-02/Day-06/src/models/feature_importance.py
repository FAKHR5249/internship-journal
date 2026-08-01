import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
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

pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", GradientBoostingRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    ))
])

print("\nTraining Model...")

pipeline.fit(X_train, y_train)

print("Calculating Permutation Importance...")

result = permutation_importance(
    pipeline,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

feature_names = X_test.columns
print(len(feature_names))
print(len(result.importances_mean))

print(feature_names[:10])
print(result.importances_mean[:10])
importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": result.importances_mean
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 20 Features")

print(importance.head(20))

importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,8))

plt.barh(
    importance.head(20)["Feature"],
    importance.head(20)["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Top 20 Feature Importance")

plt.tight_layout()

plt.savefig(
    "reports/figures/feature_importance.png",
    dpi=300
)

plt.show()

print("\nSaved:")
print("reports/feature_importance.csv")
print("reports/figures/feature_importance.png")