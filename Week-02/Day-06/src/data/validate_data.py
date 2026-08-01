import pandas as pd
from pathlib import Path

# Paths
INPUT_FILE = Path("data/processed/combined_taxi_data.parquet")
OUTPUT_FILE = Path("data/processed/clean_taxi_data.parquet")


def load_data():
    df = pd.read_parquet(INPUT_FILE)
    print("Dataset Loaded Successfully")
    return df


def create_target(df):
    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] -
        df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    return df


def validate_data(df):

    print("\n========== DATA AUDIT ==========")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nNegative Duration")
    print((df["trip_duration_minutes"] < 0).sum())

    print("\nZero Duration")
    print((df["trip_duration_minutes"] == 0).sum())

    print("\nPassenger Count <= 0")
    print((df["passenger_count"] <= 0).sum())

    print("\nTrip Distance <= 0")
    print((df["trip_distance"] <= 0).sum())


def clean_data(df):

    # Remove negative duration
    df = df[df["trip_duration_minutes"] > 0]

    # Remove invalid passenger count
    df = df[df["passenger_count"] > 0]

    # Remove zero distance
    df = df[df["trip_distance"] > 0]

    return df


def save_data(df):

    df.to_parquet(OUTPUT_FILE, index=False)

    print("\nClean Dataset Saved")
    print(OUTPUT_FILE)


def main():

    df = load_data()

    df = create_target(df)

    validate_data(df)

    df = clean_data(df)

    save_data(df)


if __name__ == "__main__":
    main()