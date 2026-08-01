# ==========================================
# Customer Personality Analysis
# Data Cleaning & Preprocessing Script
# AI Lab 99 - Module 03
# ==========================================

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("marketing_campaign.csv", sep="\t")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ==========================================
# Handle Missing Values
# ==========================================

df["Income"] = df["Income"].fillna(df["Income"].median())

# ==========================================
# Remove Duplicate Rows
# ==========================================

df.drop_duplicates(inplace=True)

# Remove Duplicate Customer IDs

df.drop_duplicates(subset="ID", keep="first", inplace=True)

# ==========================================
# Correct Data Types
# ==========================================

df["ID"] = df["ID"].astype(int)
df["Year_Birth"] = df["Year_Birth"].astype(int)
df["Income"] = df["Income"].astype(float)
df["Recency"] = df["Recency"].astype(int)

# ==========================================
# Convert Date Column
# ==========================================

df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    format="%d-%m-%Y",
    errors="coerce"
)

df["Enrollment_Year"] = df["Dt_Customer"].dt.year
df["Enrollment_Month"] = df["Dt_Customer"].dt.month
df["Enrollment_Day"] = df["Dt_Customer"].dt.day

# ==========================================
# Standardize Categories
# ==========================================

df["Education"] = (
    df["Education"]
    .str.strip()
)

df["Marital_Status"] = (
    df["Marital_Status"]
    .str.strip()
    .str.title()
)

df["Marital_Status"] = df["Marital_Status"].replace({
    "Alone": "Single",
    "Absurd": "Other",
    "Yolo": "Other"
})

# ==========================================
# Calculate Age
# ==========================================

current_year = datetime.now().year

df["Age"] = current_year - df["Year_Birth"]

# Remove unrealistic ages

df = df[
    (df["Age"] >= 0) &
    (df["Age"] <= 100)
]

# ==========================================
# Handle Outliers (Winsorization)
# ==========================================

columns = [
    "Income",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

# ==========================================
# Validation
# ==========================================

print("\nValidation Summary")

print("----------------------------")

print("Missing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

print("Duplicate IDs :", df["ID"].duplicated().sum())

print("Invalid Dates :", df["Dt_Customer"].isnull().sum())

print("\nFinal Shape :", df.shape)

# ==========================================
# Export Cleaned Dataset
# ==========================================

os.makedirs("03_Cleaned_Data", exist_ok=True)

output_path = "03_Cleaned_Data/customer_personality_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nDataset Exported Successfully")

print("Location :", output_path)