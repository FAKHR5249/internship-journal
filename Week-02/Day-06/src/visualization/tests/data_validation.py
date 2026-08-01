from pathlib import Path
import pandas as pd


def test_processed_dataset_exists():
    dataset = Path("data/processed/clean_taxi_data.parquet")
    assert dataset.exists(), "Clean dataset not found."


def test_dataset_not_empty():
    df = pd.read_parquet("data/processed/clean_taxi_data.parquet")
    assert len(df) > 0, "Dataset is empty."


def test_target_column_exists():
    df = pd.read_parquet("data/processed/clean_taxi_data.parquet")
    assert "trip_duration_minutes" in df.columns


def test_no_negative_trip_duration():
    df = pd.read_parquet("data/processed/clean_taxi_data.parquet")
    assert (df["trip_duration_minutes"] >= 0).all()


def test_pickup_before_dropoff():
    df = pd.read_parquet("data/processed/clean_taxi_data.parquet")
    assert (
        df["tpep_dropoff_datetime"] >=
        df["tpep_pickup_datetime"]
    ).all()