import pandas as pd

# Load Dataset
file_path = "01_Raw_Data/marketing_campaign.csv"
df = pd.read_csv(file_path, sep="\t")

# Duplicate Rows
duplicate_rows = df.duplicated().sum()

# Duplicate Customer IDs
duplicate_ids = df["ID"].duplicated().sum()

print("=" * 60)
print("DUPLICATE RECORDS REPORT")
print("=" * 60)

print(f"Total Records        : {len(df)}")
print(f"Duplicate Rows       : {duplicate_rows}")
print(f"Duplicate Customer IDs : {duplicate_ids}")

# Show duplicate IDs if available
if duplicate_ids > 0:
    print("\nDuplicate Customer IDs:")
    print(df[df["ID"].duplicated()]["ID"])
else:
    print("\nNo Duplicate Customer IDs Found.")