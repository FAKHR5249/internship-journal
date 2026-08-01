# ==========================================
# AI Lab 99 Internship 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 4 - Purchasing Behavior Analysis
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("marketing_campaign.csv", sep="\t")

# -----------------------------
# Plot Style
# -----------------------------
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# =====================================================
# Purchase Columns
# =====================================================

purchase_cols = [
    "NumStorePurchases",
    "NumWebPurchases",
    "NumCatalogPurchases"
]

# =====================================================
# Total Purchases
# =====================================================

df["TotalPurchases"] = df[purchase_cols].sum(axis=1)

print("=" * 60)
print("PURCHASING BEHAVIOR ANALYSIS")
print("=" * 60)

print("\nPurchase Statistics")
print(df["TotalPurchases"].describe())

# =====================================================
# Average Purchases
# =====================================================

print("\nAverage Purchases Per Customer")
print(f"{df['TotalPurchases'].mean():.2f}")

# =====================================================
# Store Purchases
# =====================================================

print("\n")
print("=" * 60)
print("STORE PURCHASES")
print("=" * 60)

print(df["NumStorePurchases"].describe())

# =====================================================
# Web Purchases
# =====================================================

print("\n")
print("=" * 60)
print("WEB PURCHASES")
print("=" * 60)

print(df["NumWebPurchases"].describe())

# =====================================================
# Catalog Purchases
# =====================================================

print("\n")
print("=" * 60)
print("CATALOG PURCHASES")
print("=" * 60)

print(df["NumCatalogPurchases"].describe())

# =====================================================
# Purchase Frequency
# =====================================================

print("\n")
print("=" * 60)
print("TOP 10 FREQUENT BUYERS")
print("=" * 60)

print(
    df[["ID", "TotalPurchases"]]
    .sort_values(by="TotalPurchases", ascending=False)
    .head(10)
)

# =====================================================
# Count Plot
# =====================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="NumStorePurchases"
)

plt.title("Store Purchase Frequency")
plt.xlabel("Store Purchases")
plt.ylabel("Customers")

plt.show()

# =====================================================
# Distribution Plot
# =====================================================

plt.figure(figsize=(9,5))

sns.histplot(
    df["TotalPurchases"],
    bins=20,
    kde=True
)

plt.title("Distribution of Total Purchases")
plt.xlabel("Total Purchases")
plt.ylabel("Customers")

plt.show()

# =====================================================
# Bar Plot
# =====================================================

purchase_totals = df[purchase_cols].sum()

plt.figure(figsize=(8,5))

sns.barplot(
    x=purchase_totals.index,
    y=purchase_totals.values
)

plt.title("Purchases by Channel")
plt.xlabel("Purchase Channel")
plt.ylabel("Total Purchases")

plt.xticks(rotation=15)

plt.show()

# =====================================================
# Box Plot
# =====================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df[purchase_cols]
)

plt.title("Purchase Distribution by Channel")

plt.show()

# =====================================================
# Top 15 Buyers
# =====================================================

top15 = (
    df.sort_values(by="TotalPurchases", ascending=False)
      .head(15)
)

plt.figure(figsize=(12,5))

sns.barplot(
    x=top15["ID"].astype(str),
    y=top15["TotalPurchases"]
)

plt.title("Top 15 Customers by Number of Purchases")
plt.xlabel("Customer ID")
plt.ylabel("Total Purchases")

plt.xticks(rotation=45)

plt.show()

# =====================================================
# Summary
# =====================================================

print("\n")
print("=" * 60)
print("PURCHASING SUMMARY")
print("=" * 60)

print(f"Average Purchases : {df['TotalPurchases'].mean():.2f}")

print(f"Maximum Purchases : {df['TotalPurchases'].max()}")

print(f"Minimum Purchases : {df['TotalPurchases'].min()}")

print("\nPurchase Channel Totals")

print(purchase_totals.sort_values(ascending=False))

print("\nMost Popular Purchase Channel :")
print(purchase_totals.idxmax())

print("\nTask 4 Completed Successfully.")