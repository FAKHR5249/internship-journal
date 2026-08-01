import numpy as np
import pandas as pd
from pathlib import Path

# ===========================
# File Paths
# ===========================

INPUT_FILE = Path("data/processed/clean_taxi_data.parquet")
LOOKUP_FILE = Path("data/raw/taxi_zone_lookup.csv")
OUTPUT_FILE = Path("data/processed/featured_taxi_data.parquet")


# ===========================
# Load Dataset
# ===========================

def load_data():
    df = pd.read_parquet(INPUT_FILE)
    print("Clean Dataset Loaded")
    return df


# ===========================
# Time Features
# ===========================

def build_time_features(df):

    # Pickup Hour
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

    # Day of Week
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek

    # Month
    df["month"] = df["tpep_pickup_datetime"].dt.month

    # Weekend
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Peak Hour
    df["is_peak_hour"] = df["pickup_hour"].isin(
        [7, 8, 9, 16, 17, 18]
    ).astype(int)

    # Night Trip
    df["is_night"] = (
        (df["pickup_hour"] >= 22) |
        (df["pickup_hour"] <= 5)
    ).astype(int)

    # Cyclical Hour Encoding
    df["hour_sin"] = np.sin(
        2 * np.pi * df["pickup_hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["pickup_hour"] / 24
    )

    # Cyclical Day Encoding
    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


# ===========================
# Geographic Features
# ===========================

def add_geographic_features(df):

    lookup = pd.read_csv(LOOKUP_FILE)

    borough_map = lookup.set_index("LocationID")["Borough"].to_dict()

    zone_map = lookup.set_index("LocationID")["Zone"].to_dict()

    # Pickup Borough
    df["pickup_borough"] = df["PULocationID"].map(borough_map)

    # Dropoff Borough
    df["dropoff_borough"] = df["DOLocationID"].map(borough_map)

    # Pickup Zone
    df["pickup_zone"] = df["PULocationID"].map(zone_map)

    # Dropoff Zone
    df["dropoff_zone"] = df["DOLocationID"].map(zone_map)

    # Same Borough
    df["same_borough"] = (
        df["pickup_borough"] ==
        df["dropoff_borough"]
    ).astype(int)

    # Airport Pickup
    airport_zones = [
        "Newark Airport",
        "JFK Airport",
        "LaGuardia Airport"
    ]

    df["airport_pickup"] = (
        df["pickup_zone"].isin(airport_zones)
    ).astype(int)

    # Airport Dropoff
    df["airport_dropoff"] = (
        df["dropoff_zone"].isin(airport_zones)
    ).astype(int)

    return df


# ===========================
# Save Dataset
# ===========================

def save_data(df):

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_FILE, index=False)

    print("\nFeatured Dataset Saved Successfully")
    print(OUTPUT_FILE)


# ===========================
# Main Function
# ===========================

def main():

    df = load_data()

    df = build_time_features(df)

    df = add_geographic_features(df)

    print("\nNew Engineered Features")

    print(
        df[
            [
                "pickup_hour",
                "day_of_week",
                "month",
                "is_weekend",
                "is_peak_hour",
                "is_night",
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
                "pickup_borough",
                "dropoff_borough",
                "pickup_zone",
                "dropoff_zone",
                "same_borough",
                "airport_pickup",
                "airport_dropoff"
            ]
        ].head()
    )
    print("\nAll Columns:")
    print(df.columns.tolist())
    save_data(df)


if __name__ == "__main__":
    main()