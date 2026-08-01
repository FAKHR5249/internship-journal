import pandas as pd

# Load Dataset
file_path = "01_Raw_Data/marketing_campaign.csv"
df = pd.read_csv(file_path, sep="\t")

print("=" * 60)
print("VARIABLE INSPECTION REPORT")
print("=" * 60)

print("\nColumn Names and Data Types:\n")

for column in df.columns:
    print(f"{column:25} {df[column].dtype}")