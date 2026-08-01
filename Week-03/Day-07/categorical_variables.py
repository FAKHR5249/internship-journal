import pandas as pd

# Load Dataset
file_path = "01_Raw_Data/marketing_campaign.csv"
df = pd.read_csv(file_path, sep="\t")

# Select Categorical Columns
categorical_columns = df.select_dtypes(include=["object", "string"]).columns

print("=" * 80)
print("CATEGORICAL VARIABLES SUMMARY")
print("=" * 80)

for col in categorical_columns:
    categories = df[col].dropna().unique()

    print(f"\nVariable : {col}")
    print(f"Number of Categories : {len(categories)}")
    print("Categories :")
    print(categories)