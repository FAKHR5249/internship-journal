# ==========================================================
# AI Lab 99 Internship Program 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 9 - Correlation Analysis
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("marketing_campaign.csv", sep="\t")

# ----------------------------------------------------------
# Feature Engineering
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
# Select Numerical Columns
# ----------------------------------------------------------
numerical_df = df.select_dtypes(include=["number"])

# ----------------------------------------------------------
# Pearson Correlation Matrix
# ----------------------------------------------------------
correlation_matrix = numerical_df.corr(method="pearson")

print("=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

print("\nCorrelation Matrix")
print("-" * 70)
print(correlation_matrix)

# ----------------------------------------------------------
# Strong Positive Correlations
# ----------------------------------------------------------
print("\n")
print("=" * 70)
print("STRONG POSITIVE CORRELATIONS (>= 0.70)")
print("=" * 70)

positive_found = False

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):

        value = correlation_matrix.iloc[i, j]

        if value >= 0.70:
            positive_found = True

            print(
                f"{correlation_matrix.columns[i]}  <-->  "
                f"{correlation_matrix.columns[j]} : {value:.3f}"
            )

if not positive_found:
    print("No strong positive correlations found.")

# ----------------------------------------------------------
# Strong Negative Correlations
# ----------------------------------------------------------
print("\n")
print("=" * 70)
print("STRONG NEGATIVE CORRELATIONS (<= -0.50)")
print("=" * 70)

negative_found = False

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):

        value = correlation_matrix.iloc[i, j]

        if value <= -0.50:
            negative_found = True

            print(
                f"{correlation_matrix.columns[i]}  <-->  "
                f"{correlation_matrix.columns[j]} : {value:.3f}"
            )

if not negative_found:
    print("No strong negative correlations found.")

# ----------------------------------------------------------
# Independent Variables
# ----------------------------------------------------------
print("\n")
print("=" * 70)
print("LIKELY INDEPENDENT VARIABLES")
print("=" * 70)

independent_variables = []

for column in correlation_matrix.columns:

    max_corr = correlation_matrix[column].drop(column).abs().max()

    if max_corr < 0.30:
        independent_variables.append(column)

if independent_variables:
    for var in independent_variables:
        print(var)
else:
    print("No independent variables found.")

# ----------------------------------------------------------
# Heatmap
# ----------------------------------------------------------
plt.figure(figsize=(18, 14))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Pearson Correlation Heatmap")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Top 10 Highest Correlations
# ----------------------------------------------------------
print("\n")
print("=" * 70)
print("TOP 10 HIGHEST CORRELATIONS")
print("=" * 70)

corr_pairs = correlation_matrix.unstack()
corr_pairs = corr_pairs[corr_pairs != 1]

corr_pairs = corr_pairs.abs().sort_values(ascending=False)
corr_pairs = corr_pairs[~corr_pairs.index.duplicated()]

print(corr_pairs.head(10))

print("\nTask 9 Completed Successfully.")