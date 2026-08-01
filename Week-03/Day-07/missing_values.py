import pandas as pd

# Load Dataset
file_path = "01_Raw_Data/marketing_campaign.csv"
df = pd.read_csv(file_path, sep="\t")

# Missing Values
missing = df.isnull().sum()

# Missing Percentage
missing_percentage = (missing / len(df)) * 100

# Create Report
report = pd.DataFrame({
    "Variable": df.columns,
    "Missing Values": missing.values,
    "Percentage": missing_percentage.values
})

print("=" * 65)
print("MISSING VALUES REPORT")
print("=" * 65)

print(report)

print("\n" + "=" * 65)
print("Columns Without Missing Values")
print("=" * 65)

no_missing = report[report["Missing Values"] == 0]

for col in no_missing["Variable"]:
    print(col)