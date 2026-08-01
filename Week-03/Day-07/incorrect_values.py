import pandas as pd
from datetime import datetime

# Load Dataset
file_path = "01_Raw_Data/marketing_campaign.csv"
df = pd.read_csv(file_path, sep="\t")

print("=" * 65)
print("INCORRECT / INCONSISTENT VALUES REPORT")
print("=" * 65)

# -----------------------------
# Invalid Birth Year
# -----------------------------
current_year = datetime.now().year

invalid_birth = df[
    (df["Year_Birth"] < 1900) |
    (df["Year_Birth"] > current_year)
]

print(f"\nInvalid Birth Years : {len(invalid_birth)}")

# -----------------------------
# Negative Income
# -----------------------------
negative_income = df[df["Income"] < 0]

print(f"Negative Income : {len(negative_income)}")

# -----------------------------
# Empty Education
# -----------------------------
empty_education = df["Education"].astype(str).str.strip().eq("").sum()

print(f"Empty Education : {empty_education}")

# -----------------------------
# Empty Marital Status
# -----------------------------
empty_marital = df["Marital_Status"].astype(str).str.strip().eq("").sum()

print(f"Empty Marital Status : {empty_marital}")

# -----------------------------
# Invalid Dates
# -----------------------------
invalid_dates = 0

for date in df["Dt_Customer"]:
    try:
        pd.to_datetime(date, format="%d-%m-%Y")
    except:
        invalid_dates += 1

print(f"Invalid Dates : {invalid_dates}")

# -----------------------------
# Education Categories
# -----------------------------
print("\nEducation Categories:")
print(df["Education"].unique())

# -----------------------------
# Marital Status Categories
# -----------------------------
print("\nMarital Status Categories:")
print(df["Marital_Status"].unique())