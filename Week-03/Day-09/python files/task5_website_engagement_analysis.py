# ==========================================
# AI Lab 99 Internship 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 5 - Website Engagement Analysis
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
# Website Visits Analysis
# ===================================================

print("="*60)
print("WEBSITE ENGAGEMENT ANALYSIS")
print("="*60)

print("\nWebsite Visits Statistics")
print(df["NumWebVisitsMonth"].describe())

# ===================================================
# Customer Engagement
# ===================================================

print("\n")
print("="*60)
print("CUSTOMER ENGAGEMENT LEVEL")
print("="*60)

def engagement(visits):
    if visits <= 3:
        return "High Engagement"
    elif visits <= 6:
        return "Medium Engagement"
    else:
        return "Low Engagement"

df["Engagement"] = df["NumWebVisitsMonth"].apply(engagement)

engagement_counts = df["Engagement"].value_counts()

print(engagement_counts)

# ===================================================
# Frequently Visiting Customers
# ===================================================

print("\n")
print("="*60)
print("TOP 10 FREQUENT WEBSITE VISITORS")
print("="*60)

top_visitors = df.sort_values(
    by="NumWebVisitsMonth",
    ascending=False
)[["ID", "NumWebVisitsMonth"]].head(10)

print(top_visitors)

# ===================================================
# Low Engagement Customers
# ===================================================

print("\n")
print("="*60)
print("LOW ENGAGEMENT CUSTOMERS")
print("="*60)

low_engagement = df[df["Engagement"] == "Low Engagement"]

print(f"Number of Low Engagement Customers : {len(low_engagement)}")

# ===================================================
# Histogram
# ===================================================

plt.figure(figsize=(9,5))

sns.histplot(
    df["NumWebVisitsMonth"],
    bins=12,
    kde=True
)

plt.title("Distribution of Website Visits")
plt.xlabel("Website Visits Per Month")
plt.ylabel("Customers")

plt.show()

# ===================================================
# Scatter Plot
# ===================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="NumWebVisitsMonth",
    y="NumWebPurchases"
)

plt.title("Website Visits vs Web Purchases")
plt.xlabel("Website Visits")
plt.ylabel("Web Purchases")

plt.show()

# ===================================================
# Density Plot
# ===================================================

plt.figure(figsize=(9,5))

sns.kdeplot(
    data=df,
    x="NumWebVisitsMonth",
    fill=True
)

plt.title("Density Plot of Website Visits")
plt.xlabel("Website Visits")

plt.show()

# ===================================================
# Engagement Bar Chart
# ===================================================

plt.figure(figsize=(7,5))

sns.barplot(
    x=engagement_counts.index,
    y=engagement_counts.values
)

plt.title("Customer Engagement Levels")
plt.xlabel("Engagement")
plt.ylabel("Customers")

plt.show()

# ===================================================
# Summary
# ===================================================

print("\n")
print("="*60)
print("WEBSITE ENGAGEMENT SUMMARY")
print("="*60)

print(f"Average Website Visits : {df['NumWebVisitsMonth'].mean():.2f}")

print(f"Maximum Website Visits : {df['NumWebVisitsMonth'].max()}")

print(f"Minimum Website Visits : {df['NumWebVisitsMonth'].min()}")

print("\nEngagement Levels")

print(engagement_counts)

print("\nTask 5 Completed Successfully.")