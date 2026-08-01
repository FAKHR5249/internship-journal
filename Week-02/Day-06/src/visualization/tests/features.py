from pathlib import Path
import pandas as pd


def test_featured_dataset_exists():
    dataset = Path("data/processed/featured_taxi_data.parquet")
    assert dataset.exists(), "Featured dataset not found."


def test_time_features_exist():
    df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

    expected = [
        "pickup_hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_peak_hour",
        "is_night",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos"
    ]

    for col in expected:
        assert col in df.columns


def test_geographic_features_exist():
    df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

    expected = [
        "pickup_borough",
        "dropoff_borough",
        "pickup_zone",
        "dropoff_zone",
        "same_borough",
        "airport_pickup",
        "airport_dropoff"
    ]

    for col in expected:
        assert col in df.columns


def test_no_missing_pickup_hour():
    df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

    assert df["pickup_hour"].isnull().sum() == 0


def test_weekend_values():
    df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_peak_hour_values():
    df = pd.read_parquet("data/processed/featured_taxi_data.parquet")

    assert set(df["is_peak_hour"].unique()).issubset({0, 1})