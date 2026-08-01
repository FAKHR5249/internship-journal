import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")


def load_monthly_files():

    files = sorted(RAW_DATA_PATH.glob("yellow_tripdata_2023-*.parquet"))

    if len(files) == 0:
        raise FileNotFoundError("No parquet files found!")

    dataframes = []

    for file in files:
        print(f"Loading {file.name}")

        df = pd.read_parquet(file).sample(
    n=100000,
    random_state=42
)

        df["source_file"] = file.name

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


def basic_information(df):

    print("\nDataset Shape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())


def save_dataset(df):

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DATA_PATH / "combined_taxi_data.parquet"

    df.to_parquet(output_file, index=False)

    print(f"\nDataset Saved Successfully")
    print(output_file)


def main():

    df = load_monthly_files()

    basic_information(df)

    save_dataset(df)


if __name__ == "__main__":
    main()