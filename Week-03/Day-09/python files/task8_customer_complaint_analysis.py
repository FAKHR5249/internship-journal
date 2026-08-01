# ==========================================================
# AI Lab 99 Internship Program 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 8 - Customer Complaint Analysis
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("marketing_campaign.csv", sep="\t")

sns.set_style("whitegrid")

# ----------------------------------------------------------
# Total Spending
# ----------------------------------------------------------
spending_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

df["Total_Spending"] = df[spending_columns].sum(axis=1)

# ----------------------------------------------------------
# Complaint Frequency
# ----------------------------------------------------------
print("=" * 60)
print("CUSTOMER COMPLAINT ANALYSIS")
print("=" * 60)

complaint_counts = df["Complain"].value_counts().sort_index()

print("\nComplaint Frequency")
print("-" * 60)

print(f"No Complaint (0) : {complaint_counts.get(0,0)}")
print(f"Complaint (1)    : {complaint_counts.get(1,0)}")

# ----------------------------------------------------------
# Complaint Percentage
# ----------------------------------------------------------
print("\nComplaint Percentage")
print("-" * 60)

complaint_percentage = (
    complaint_counts / len(df)
) * 100

print(complaint_percentage.round(2))

# ----------------------------------------------------------
# Complaints vs Spending
# ----------------------------------------------------------
print("\n")
print("=" * 60)
print("COMPLAINTS VS SPENDING")
print("=" * 60)

spending_analysis = (
    df.groupby("Complain")["Total_Spending"]
      .agg(["count", "mean", "median", "max", "min"])
)

print(spending_analysis)

# ----------------------------------------------------------
# Complaints vs Income
# ----------------------------------------------------------
print("\n")
print("=" * 60)
print("COMPLAINTS VS INCOME")
print("=" * 60)

income_analysis = (
    df.groupby("Complain")["Income"]
      .agg(["count", "mean", "median", "max", "min"])
)

print(income_analysis)

# ----------------------------------------------------------
# Count Plot
# ----------------------------------------------------------
plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Complain"
)

plt.title("Customer Complaints")
plt.xlabel("Complaint (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.show()

# ----------------------------------------------------------
# Box Plot (Complaints vs Spending)
# ----------------------------------------------------------
plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Complain",
    y="Total_Spending"
)

plt.title("Complaints vs Total Spending")
plt.xlabel("Complaint")
plt.ylabel("Total Spending")

plt.show()

# ----------------------------------------------------------
# Box Plot (Complaints vs Income)
# ----------------------------------------------------------
plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Complain",
    y="Income"
)

plt.title("Complaints vs Income")
plt.xlabel("Complaint")
plt.ylabel("Income")

plt.show()

# ----------------------------------------------------------
# Correlation Heatmap
# ----------------------------------------------------------
plt.figure(figsize=(7,5))

corr = df[
    ["Complain", "Income", "Total_Spending"]
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Complaint Correlation Heatmap")

plt.show()

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
print("\n")
print("=" * 60)
print("CUSTOMER COMPLAINT SUMMARY")
print("=" * 60)

print(f"Total Customers : {len(df)}")

print(f"Customers with Complaints : {complaint_counts.get(1,0)}")

print(f"Customers without Complaints : {complaint_counts.get(0,0)}")

print(f"Complaint Rate : {complaint_percentage.get(1,0):.2f}%")

print(f"\nAverage Spending (Complaint Customers) : "
      f"{spending_analysis.loc[1,'mean']:.2f}")

print(f"Average Spending (No Complaint Customers) : "
      f"{spending_analysis.loc[0,'mean']:.2f}")

print(f"\nAverage Income (Complaint Customers) : "
      f"{income_analysis.loc[1,'mean']:.2f}")

print(f"Average Income (No Complaint Customers) : "
      f"{income_analysis.loc[0,'mean']:.2f}")

print("\nTask 8 Completed Successfully.")