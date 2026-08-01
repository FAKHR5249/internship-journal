# ==========================================
# AI Lab 99 Internship 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 3 - Customer Spending Analysis
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
plt.rcParams["figure.figsize"] = (10,6)

# ===================================================
# Spending Columns
# ===================================================

spending_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

# ===================================================
# Total Spending
# ===================================================

df["Total_Spending"] = df[spending_cols].sum(axis=1)

print("="*60)
print("TOTAL SPENDING ANALYSIS")
print("="*60)

print(df["Total_Spending"].describe())

print("\nTop 10 Highest Spending Customers")
print(df[["ID","Total_Spending"]]
      .sort_values(by="Total_Spending", ascending=False)
      .head(10))

# ===================================================
# Average Spending
# ===================================================

print("\n")
print("="*60)
print("AVERAGE SPENDING")
print("="*60)

average_spending = df["Total_Spending"].mean()

print(f"Average Customer Spending : {average_spending:.2f}")

# ===================================================
# Spending by Product Category
# ===================================================

print("\n")
print("="*60)
print("PRODUCT CATEGORY SPENDING")
print("="*60)

category_spending = df[spending_cols].sum()

print(category_spending)

# ===================================================
# Histogram
# ===================================================

plt.figure(figsize=(9,5))

sns.histplot(
    df["Total_Spending"],
    bins=30,
    kde=True
)

plt.title("Distribution of Total Customer Spending")
plt.xlabel("Total Spending")
plt.ylabel("Customers")
plt.show()

# ===================================================
# Box Plot
# ===================================================

plt.figure(figsize=(9,2))

sns.boxplot(
    x=df["Total_Spending"]
)

plt.title("Box Plot of Total Spending")
plt.show()

# ===================================================
# Product Category Bar Chart
# ===================================================

plt.figure(figsize=(10,5))

sns.barplot(
    x=category_spending.index,
    y=category_spending.values
)

plt.title("Total Spending by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Spending")

plt.xticks(rotation=20)

plt.show()

# ===================================================
# Spending by Customer (Top 15)
# ===================================================

top15 = df.sort_values(
    by="Total_Spending",
    ascending=False
).head(15)

plt.figure(figsize=(12,5))

sns.barplot(
    x=top15["ID"].astype(str),
    y=top15["Total_Spending"]
)

plt.title("Top 15 Highest Spending Customers")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")

plt.xticks(rotation=45)

plt.show()

# ===================================================
# Stacked Bar Chart
# ===================================================

top10 = df.sort_values(
    by="Total_Spending",
    ascending=False
).head(10)

top10_plot = top10.set_index("ID")[spending_cols]

top10_plot.plot(
    kind="bar",
    stacked=True,
    figsize=(12,6)
)

plt.title("Top 10 Customers Spending by Product Category")
plt.xlabel("Customer ID")
plt.ylabel("Amount Spent")

plt.legend(title="Category")

plt.tight_layout()
plt.show()

# ===================================================
# Product Category Box Plot
# ===================================================

plt.figure(figsize=(11,6))

sns.boxplot(
    data=df[spending_cols]
)

plt.title("Product Category Spending Distribution")

plt.xticks(rotation=20)

plt.show()

# ===================================================
# Summary
# ===================================================

print("\n")
print("="*60)
print("CUSTOMER SPENDING SUMMARY")
print("="*60)

print(f"Average Spending : {df['Total_Spending'].mean():.2f}")

print(f"Median Spending  : {df['Total_Spending'].median():.2f}")

print(f"Maximum Spending : {df['Total_Spending'].max():.2f}")

print(f"Minimum Spending : {df['Total_Spending'].min():.2f}")

print("\nHighest Revenue Product Category")

print(category_spending.sort_values(ascending=False))

print("\nTask 3 Completed Successfully.")