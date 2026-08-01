# ==========================================================
# Build Dataset
# ==========================================================

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")
OUTPUT_PATH = Path("data/processed/clean_taxi_data.parquet")


def load_all_files():

    files = list(
        RAW_PATH.glob("yellow_tripdata_*.parquet")
    )

    print(f"Files Found: {len(files)}")

    dataframes = []

    for file in files:

        print(f"Loading: {file.name}")

        df = pd.read_parquet(file)

        df["source_file"] = file.name

        dataframes.append(df)

    final_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return final_df


def clean_data(df):

    print("\nCleaning Dataset")

    # Create target
    df["trip_duration_minutes"] = (
        (
            df["tpep_dropoff_datetime"] -
            df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60
    )

    # Remove invalid trips
    df = df[
        (df["trip_duration_minutes"] > 1) &
        (df["trip_duration_minutes"] < 180)
    ]

    # Passenger count
    df = df[
        df["passenger_count"] > 0
    ]

    # Distance
    if "trip_distance" in df.columns:

        df = df[
            df["trip_distance"] > 0
        ]

    print(f"Final Shape: {df.shape}")

    return df


def save_data(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("\nDataset Saved")

    print(OUTPUT_PATH)


def main():

    df = load_all_files()

    df = clean_data(df)

    save_data(df)


if __name__ == "__main__":
    main()