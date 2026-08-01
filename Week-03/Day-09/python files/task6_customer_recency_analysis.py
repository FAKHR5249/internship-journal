# ==========================================
# AI Lab 99 Internship 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 6 - Customer Recency Analysis
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
plt.rcParams["figure.figsize"] = (10,5)

# ===================================================
# Recency Statistics
# ===================================================

print("="*60)
print("CUSTOMER RECENCY ANALYSIS")
print("="*60)

print("\nRecency Statistics")
print(df["Recency"].describe())

# ===================================================
# Customer Classification
# ===================================================
# Lower Recency = More Active Customer

def customer_status(recency):

    if recency <= 30:
        return "Active"

    elif recency <= 60:
        return "Moderately Active"

    else:
        return "Inactive"


df["Customer_Status"] = df["Recency"].apply(customer_status)

status_counts = df["Customer_Status"].value_counts()

# ===================================================
# Active Customers
# ===================================================

print("\n")
print("="*60)
print("CUSTOMER STATUS")
print("="*60)

print(status_counts)

# ===================================================
# Top Active Customers
# ===================================================

print("\n")
print("="*60)
print("TOP 10 MOST RECENT CUSTOMERS")
print("="*60)

active_customers = df.sort_values("Recency")

print(
    active_customers[
        ["ID","Recency"]
    ].head(10)
)

# ===================================================
# Inactive Customers
# ===================================================

print("\n")
print("="*60)
print("TOP 10 LEAST RECENT CUSTOMERS")
print("="*60)

inactive_customers = df.sort_values(
    "Recency",
    ascending=False
)

print(
    inactive_customers[
        ["ID","Recency"]
    ].head(10)
)

print("\nTotal Inactive Customers:",
      len(df[df["Customer_Status"]=="Inactive"]))

# ===================================================
# Histogram
# ===================================================

plt.figure(figsize=(9,5))

sns.histplot(
    df["Recency"],
    bins=20,
    kde=True
)

plt.title("Distribution of Customer Recency")
plt.xlabel("Recency (Days)")
plt.ylabel("Customers")

plt.show()

# ===================================================
# Box Plot
# ===================================================

plt.figure(figsize=(8,2))

sns.boxplot(
    x=df["Recency"]
)

plt.title("Customer Recency Box Plot")

plt.show()

# ===================================================
# Customer Status Bar Chart
# ===================================================

plt.figure(figsize=(7,5))

sns.barplot(
    x=status_counts.index,
    y=status_counts.values
)

plt.title("Customer Activity Status")
plt.xlabel("Status")
plt.ylabel("Customers")

plt.show()

# ===================================================
# Pie Chart
# ===================================================

plt.figure(figsize=(7,7))

plt.pie(
    status_counts.values,
    labels=status_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Customer Activity Distribution")

plt.show()

# ===================================================
# Recency vs Spending
# ===================================================

spending_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

df["Total_Spending"] = df[spending_cols].sum(axis=1)

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="Recency",
    y="Total_Spending"
)

plt.title("Recency vs Total Spending")
plt.xlabel("Recency")
plt.ylabel("Total Spending")

plt.show()

# ===================================================
# Summary
# ===================================================

print("\n")
print("="*60)
print("RECENCY SUMMARY")
print("="*60)

print(f"Average Recency : {df['Recency'].mean():.2f} Days")
print(f"Minimum Recency : {df['Recency'].min()} Days")
print(f"Maximum Recency : {df['Recency'].max()} Days")

print("\nCustomer Status")

print(status_counts)

print("\nCustomers Needing Re-engagement")

print(df[df["Customer_Status"]=="Inactive"][["ID","Recency"]].head(20))

print("\nTask 6 Completed Successfully.")