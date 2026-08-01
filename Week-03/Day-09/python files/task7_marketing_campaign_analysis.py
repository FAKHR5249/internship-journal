# ==========================================
# AI Lab 99 Internship 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 7 - Marketing Campaign Analysis
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

print("="*60)
print("MARKETING CAMPAIGN ANALYSIS")
print("="*60)

# ===================================================
# Campaign Columns
# ===================================================

campaigns = {
    "Campaign 1": "AcceptedCmp1",
    "Campaign 2": "AcceptedCmp2",
    "Campaign 3": "AcceptedCmp3",
    "Campaign 4": "AcceptedCmp4",
    "Campaign 5": "AcceptedCmp5",
    "Latest Campaign": "Response"
}

# ===================================================
# Acceptance Rate
# ===================================================

print("\nCampaign Performance")
print("-"*60)

campaign_summary = []

for name, column in campaigns.items():

    accepted = df[column].sum()

    rejected = len(df) - accepted

    acceptance_rate = (accepted / len(df)) * 100

    campaign_summary.append([
        name,
        accepted,
        rejected,
        round(acceptance_rate,2)
    ])

summary_df = pd.DataFrame(
    campaign_summary,
    columns=[
        "Campaign",
        "Accepted",
        "Rejected",
        "Acceptance Rate (%)"
    ]
)

print(summary_df)

# ===================================================
# Response Rate
# ===================================================

print("\n")
print("="*60)
print("RESPONSE RATE")
print("="*60)

latest_response = (df["Response"].sum()/len(df))*100

print(f"Latest Campaign Response Rate : {latest_response:.2f}%")

# ===================================================
# Best Campaign
# ===================================================

best_campaign = summary_df.loc[
    summary_df["Acceptance Rate (%)"].idxmax()
]

print("\nBest Performing Campaign")
print(best_campaign)

# ===================================================
# Count Plot (Latest Campaign)
# ===================================================

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="Response"
)

plt.title("Latest Campaign Response")
plt.xlabel("Response (0 = No, 1 = Yes)")
plt.ylabel("Customers")

plt.show()

# ===================================================
# Bar Chart
# ===================================================

plt.figure(figsize=(10,5))

sns.barplot(
    data=summary_df,
    x="Campaign",
    y="Acceptance Rate (%)"
)

plt.title("Campaign Acceptance Rate")
plt.xlabel("Campaign")
plt.ylabel("Acceptance Rate (%)")

plt.show()

# ===================================================
# Pie Chart
# ===================================================

plt.figure(figsize=(8,8))

plt.pie(
    summary_df["Accepted"],
    labels=summary_df["Campaign"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Accepted Customers by Campaign")

plt.show()

# ===================================================
# Campaign Success Ranking
# ===================================================

print("\n")
print("="*60)
print("CAMPAIGN SUCCESS RANKING")
print("="*60)

ranking = summary_df.sort_values(
    by="Acceptance Rate (%)",
    ascending=False
)

print(ranking)

# ===================================================
# Summary
# ===================================================

print("\n")
print("="*60)
print("MARKETING CAMPAIGN SUMMARY")
print("="*60)

print(f"Highest Acceptance Rate : {ranking.iloc[0]['Campaign']}")
print(f"Acceptance Rate         : {ranking.iloc[0]['Acceptance Rate (%)']}%")

print(f"\nLowest Acceptance Rate  : {ranking.iloc[-1]['Campaign']}")
print(f"Acceptance Rate         : {ranking.iloc[-1]['Acceptance Rate (%)']}%")

print(f"\nOverall Latest Response Rate : {latest_response:.2f}%")

print("\nTask 7 Completed Successfully.")