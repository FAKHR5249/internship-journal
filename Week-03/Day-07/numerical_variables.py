import pandas as pd

# Load Dataset
file_path = "01_Raw_Data/marketing_campaign.csv"
df = pd.read_csv(file_path, sep="\t")

# Select Numerical Columns
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

# Create Summary Table
summary = pd.DataFrame({
    "Variable": numerical_columns,
    "Minimum": [df[col].min() for col in numerical_columns],
    "Maximum": [df[col].max() for col in numerical_columns],
    "Mean": [round(df[col].mean(), 2) for col in numerical_columns]
})

print("=" * 80)
print("NUMERICAL VARIABLES SUMMARY")
print("=" * 80)

print(summary.to_string(index=False))