# ==========================================
# AI Lab 99 Internship 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 1: Dataset Overview
# ==========================================

import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
file_path = "marketing_campaign.csv"

# Use tab separator if required
df = pd.read_csv(file_path, sep="\t")

print("=" * 70)
print("CUSTOMER PERSONALITY ANALYSIS DATASET")
print("=" * 70)

# -----------------------------
# Dataset Dimensions
# -----------------------------
print("\n1. Dataset Dimensions")
print("-" * 40)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# -----------------------------
# Column Names
# -----------------------------
print("\n2. Column Names")
print("-" * 40)

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

# -----------------------------
# Data Types
# -----------------------------
print("\n3. Data Types")
print("-" * 40)
print(df.dtypes)

# -----------------------------
# Dataset Information
# -----------------------------
print("\n4. Dataset Information")
print("-" * 40)
df.info()

# -----------------------------
# Descriptive Statistics
# -----------------------------
print("\n5. Descriptive Statistics")
print("-" * 40)
print(df.describe())

# -----------------------------
# Statistics Including Categorical Columns
# -----------------------------
print("\n6. Complete Statistics")
print("-" * 40)
print(df.describe(include='all'))

# -----------------------------
# Duplicate Records
# -----------------------------
duplicates = df.duplicated().sum()

print("\n7. Duplicate Records")
print("-" * 40)
print(f"Duplicate Rows : {duplicates}")

# -----------------------------
# Missing Values
# -----------------------------
missing = df.isnull().sum()

print("\n8. Missing Values")
print("-" * 40)
print(missing)

missing_percent = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Values": missing,
    "Percentage (%)": missing_percent.round(2)
})

print("\n9. Missing Value Report")
print("-" * 40)
print(missing_report)

# -----------------------------
# Dataset Summary
# -----------------------------
print("\n10. Dataset Summary")
print("-" * 40)

print(f"Total Rows            : {df.shape[0]}")
print(f"Total Columns         : {df.shape[1]}")
print(f"Numerical Columns     : {len(df.select_dtypes(include=['int64','float64']).columns)}")
print(f"Categorical Columns   : {len(df.select_dtypes(include=['object', 'str']).columns)}")
print(f"Duplicate Rows        : {duplicates}")
print(f"Columns with Missing Values : {(missing > 0).sum()}")

print("\nEDA Task 1 Completed Successfully.")