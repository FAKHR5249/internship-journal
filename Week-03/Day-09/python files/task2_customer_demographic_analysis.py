# ==========================================
# AI Lab 99 Internship 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 2 - Customer Demographic Analysis
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("marketing_campaign.csv", sep="\t")

# Style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# -----------------------------
# Create Age Column
# -----------------------------
CURRENT_YEAR = 2026
df["Age"] = CURRENT_YEAR - df["Year_Birth"]

# ===================================================
# 1. Age Distribution
# ===================================================

print("="*60)
print("AGE DISTRIBUTION")
print("="*60)

print(df["Age"].describe())

plt.figure()
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

plt.figure()
sns.boxplot(x=df["Age"])
plt.title("Age Box Plot")
plt.show()

# ===================================================
# 2. Income Distribution
# ===================================================

print("\n")
print("="*60)
print("INCOME DISTRIBUTION")
print("="*60)

print(df["Income"].describe())

plt.figure()
sns.histplot(df["Income"], bins=30, kde=True)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Customers")
plt.show()

plt.figure()
sns.boxplot(x=df["Income"])
plt.title("Income Box Plot")
plt.show()

# ===================================================
# 3. Education Distribution
# ===================================================

print("\n")
print("="*60)
print("EDUCATION LEVEL")
print("="*60)

education_counts = df["Education"].value_counts()

print(education_counts)

plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="Education",
    order=education_counts.index
)

plt.title("Education Level Distribution")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(7,7))
plt.pie(
    education_counts,
    labels=education_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Education Distribution")
plt.show()

# ===================================================
# 4. Marital Status
# ===================================================

print("\n")
print("="*60)
print("MARITAL STATUS")
print("="*60)

marital_counts = df["Marital_Status"].value_counts()

print(marital_counts)

plt.figure(figsize=(9,5))
sns.countplot(
    data=df,
    x="Marital_Status",
    order=marital_counts.index
)

plt.title("Marital Status Distribution")
plt.xticks(rotation=35)
plt.show()

plt.figure(figsize=(8,8))
plt.pie(
    marital_counts,
    labels=marital_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Marital Status Distribution")
plt.show()

# ===================================================
# 5. Household Composition
# ===================================================

print("\n")
print("="*60)
print("HOUSEHOLD COMPOSITION")
print("="*60)

df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]

household = df["TotalChildren"].value_counts().sort_index()

print(household)

plt.figure(figsize=(7,5))
sns.countplot(
    data=df,
    x="TotalChildren"
)

plt.title("Household Composition")
plt.xlabel("Children in Household")
plt.ylabel("Customers")
plt.show()

# ===================================================
# 6. Number of Children
# ===================================================

print("\n")
print("="*60)
print("KIDHOME")
print("="*60)

print(df["Kidhome"].value_counts().sort_index())

plt.figure(figsize=(6,4))
sns.countplot(
    data=df,
    x="Kidhome"
)

plt.title("Number of Children")
plt.show()

# ===================================================
# 7. Number of Teenagers
# ===================================================

print("\n")
print("="*60)
print("TEENHOME")
print("="*60)

print(df["Teenhome"].value_counts().sort_index())

plt.figure(figsize=(6,4))
sns.countplot(
    data=df,
    x="Teenhome"
)

plt.title("Number of Teenagers")
plt.show()

# ===================================================
# Summary
# ===================================================

print("\n")
print("="*60)
print("DEMOGRAPHIC SUMMARY")
print("="*60)

print(f"Average Age           : {df['Age'].mean():.2f} Years")
print(f"Minimum Age           : {df['Age'].min()} Years")
print(f"Maximum Age           : {df['Age'].max()} Years")

print(f"\nAverage Income        : {df['Income'].mean():,.2f}")
print(f"Median Income         : {df['Income'].median():,.2f}")

print(f"\nMost Common Education : {education_counts.idxmax()}")
print(f"Most Common Marital Status : {marital_counts.idxmax()}")

print(f"\nAverage Children      : {df['Kidhome'].mean():.2f}")
print(f"Average Teenagers     : {df['Teenhome'].mean():.2f}")

print("\nTask 2 Completed Successfully.")