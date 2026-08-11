# ==========================================================
# Module 7: Cluster Evaluation & Customer Profiling
# Activity 1: Cluster Statistics & Demographic Analysis
# Author: Fakhr Ul Islam
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Create Output Folders
# ==========================================================

os.makedirs("reports", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("=" * 60)
print("MODULE 7 - ACTIVITY 1")
print("Cluster Statistics & Demographic Analysis")
print("=" * 60)

# ==========================================================
# Load Datasets
# ==========================================================

original_df = pd.read_csv("customer_personality_cleaned.csv")
cluster_df = pd.read_csv("customer_clusters.csv")

# Add Cluster Column
original_df["Cluster"] = cluster_df["Cluster"]

print("\nDatasets Loaded Successfully")
print(f"Original Dataset : {original_df.shape}")
print(f"Cluster Dataset  : {cluster_df.shape}")

# ==========================================================
# Cluster Statistics
# ==========================================================

print("\nCalculating Cluster Statistics...")

cluster_statistics = original_df.groupby("Cluster").mean(numeric_only=True).round(2)

cluster_statistics.to_excel(
    "reports/module7_cluster_statistics.xlsx"
)

print("Cluster Statistics Saved")

# ==========================================================
# Cluster Size
# ==========================================================

cluster_size = original_df["Cluster"].value_counts().sort_index()

cluster_size.to_excel(
    "reports/module7_cluster_size.xlsx"
)

print("Cluster Size Saved")

print(cluster_size)
print("\n" + "=" * 60)
print("DEMOGRAPHIC ANALYSIS")
print("=" * 60)

# ==========================================================
# Customer Age
# ==========================================================

original_df["Customer_Age"] = original_df["Age"]

age_comparison = (
    original_df
    .groupby("Cluster")["Customer_Age"]
    .mean()
    .round(2)
)

print("\nAverage Customer Age")
print(age_comparison)

age_comparison.to_excel(
    "reports/age_comparison.xlsx"
)

plt.figure(figsize=(8,5))
plt.bar(age_comparison.index.astype(str), age_comparison.values)
plt.title("Average Customer Age by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Age")
plt.tight_layout()
plt.savefig("figures/age_comparison.png")
plt.close()

# ==========================================================
# Income Comparison
# ==========================================================

income_comparison = (
    original_df
    .groupby("Cluster")["Income"]
    .mean()
    .round(2)
)

print("\nAverage Income")
print(income_comparison)

income_comparison.to_excel(
    "reports/income_comparison.xlsx"
)

plt.figure(figsize=(8,5))
plt.bar(income_comparison.index.astype(str), income_comparison.values)
plt.title("Average Income by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Income")
plt.tight_layout()
plt.savefig("figures/income_comparison.png")
plt.close()

# ==========================================================
# Education Distribution
# ==========================================================

education_distribution = pd.crosstab(
    original_df["Cluster"],
    original_df["Education"]
)

print("\nEducation Distribution")
print(education_distribution)

education_distribution.to_excel(
    "reports/education_distribution.xlsx"
)

education_distribution.plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Education Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("figures/education_distribution.png")
plt.close()

# ==========================================================
# Marital Status Distribution
# ==========================================================

marital_distribution = pd.crosstab(
    original_df["Cluster"],
    original_df["Marital_Status"]
)

print("\nMarital Status Distribution")
print(marital_distribution)

marital_distribution.to_excel(
    "reports/marital_distribution.xlsx"
)

marital_distribution.plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Marital Status Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("figures/marital_distribution.png")
plt.close()

# ==========================================================
# Family Size Comparison
# ==========================================================

original_df["Family_Size"] = (
    original_df["Kidhome"] +
    original_df["Teenhome"] +
    2
)

family_size = (
    original_df
    .groupby("Cluster")["Family_Size"]
    .mean()
    .round(2)
)

print("\nAverage Family Size")
print(family_size)

# ==========================================================
# Children Comparison
# ==========================================================

original_df["Children"] = (
    original_df["Kidhome"] +
    original_df["Teenhome"]
)

children = (
    original_df
    .groupby("Cluster")["Children"]
    .mean()
    .round(2)
)

print("\nAverage Children")
print(children)

# ==========================================================
# Final Demographic Comparison Table
# ==========================================================

demographic_table = pd.DataFrame({
    "Customers": cluster_size,
    "Average Age": age_comparison,
    "Average Income": income_comparison,
    "Average Family Size": family_size,
    "Average Children": children
})

print("\nDemographic Comparison Table")
print(demographic_table)

demographic_table.to_excel(
    "reports/demographic_comparison.xlsx"
)

print("\nActivity 1 Completed Successfully")

# ==========================================================
# Feature Engineering for Module 7
# ==========================================================

print("\nCreating Required Features...")

# Total Spending
original_df["Total_Spending"] = (
    original_df["MntWines"] +
    original_df["MntFruits"] +
    original_df["MntMeatProducts"] +
    original_df["MntFishProducts"] +
    original_df["MntSweetProducts"] +
    original_df["MntGoldProds"]
)

# Family Size
original_df["Family_Size"] = (
    original_df["Kidhome"] +
    original_df["Teenhome"] +
    2
)

# Total Children
original_df["Total_Children"] = (
    original_df["Kidhome"] +
    original_df["Teenhome"]
)

# Purchase Frequency
original_df["Purchase_Frequency"] = (
    original_df["NumWebPurchases"] +
    original_df["NumStorePurchases"] +
    original_df["NumCatalogPurchases"]
)

# Accepted Campaigns
original_df["Accepted_Campaigns"] = (
    original_df["AcceptedCmp1"] +
    original_df["AcceptedCmp2"] +
    original_df["AcceptedCmp3"] +
    original_df["AcceptedCmp4"] +
    original_df["AcceptedCmp5"] +
    original_df["Response"]
)

print("Feature Engineering Completed Successfully")
# ==========================================================
# Activity 2 : Spending Behavior Analysis
# ==========================================================

print("\n" + "=" * 60)
print("ACTIVITY 2 : SPENDING BEHAVIOR ANALYSIS")
print("=" * 60)

# ==========================================================
# Total Spending
# ==========================================================

total_spending = (
    original_df
    .groupby("Cluster")["Total_Spending"]
    .mean()
    .round(2)
)

print("\nAverage Total Spending")
print(total_spending)

total_spending.to_excel(
    "reports/total_spending.xlsx"
)

plt.figure(figsize=(8,5))
plt.bar(total_spending.index.astype(str), total_spending.values)

plt.title("Average Total Spending by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Spending")

plt.tight_layout()
plt.savefig("figures/total_spending.png")
plt.close()

# ==========================================================
# Product Spending Comparison
# ==========================================================

product_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

product_spending = (
    original_df
    .groupby("Cluster")[product_columns]
    .mean()
    .round(2)
)

print("\nProduct Spending")
print(product_spending)

product_spending.to_excel(
    "reports/product_spending.xlsx"
)

# ==========================================================
# Product Spending Chart
# ==========================================================

product_spending.plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Average Product Spending by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Spending")

plt.tight_layout()
plt.savefig("figures/product_spending.png")
plt.close()

# ==========================================================
# Highest Spending Cluster
# ==========================================================

highest_cluster = total_spending.idxmax()
highest_value = total_spending.max()

print("\nHighest Spending Cluster")
print(f"Cluster : {highest_cluster}")
print(f"Average Spending : {highest_value:.2f}")

# ==========================================================
# Lowest Spending Cluster
# ==========================================================

lowest_cluster = total_spending.idxmin()
lowest_value = total_spending.min()

print("\nLowest Spending Cluster")
print(f"Cluster : {lowest_cluster}")
print(f"Average Spending : {lowest_value:.2f}")

# ==========================================================
# Premium Product Buyers
# ==========================================================

premium_products = (
    product_spending[
        ["MntWines", "MntMeatProducts"]
    ].sum(axis=1)
)

premium_cluster = premium_products.idxmax()

print("\nPremium Product Buyers")
print(f"Cluster : {premium_cluster}")

# ==========================================================
# Budget Conscious Customers
# ==========================================================

budget_cluster = total_spending.idxmin()

print("\nBudget Conscious Customers")
print(f"Cluster : {budget_cluster}")

# ==========================================================
# Product Preference Summary
# ==========================================================

preference_summary = product_spending.idxmax(axis=1)

summary = pd.DataFrame({
    "Preferred_Product": preference_summary
})

print("\nProduct Preference Summary")
print(summary)

summary.to_excel(
    "reports/product_preference_summary.xlsx"
)

print("\nActivity 2 Completed Successfully")
# ==========================================================
# Activity 3 : Shopping Channel & Customer Engagement Analysis
# ==========================================================

print("\n" + "=" * 60)
print("ACTIVITY 3 : SHOPPING CHANNEL & CUSTOMER ENGAGEMENT")
print("=" * 60)

# ==========================================================
# Channel Comparison
# ==========================================================

channel_columns = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    "Recency"
]

channel_analysis = (
    original_df
    .groupby("Cluster")[channel_columns]
    .mean()
    .round(2)
)

print("\nShopping Channel Analysis")
print(channel_analysis)

channel_analysis.to_excel(
    "reports/channel_analysis.xlsx"
)

# ==========================================================
# Shopping Channel Chart
# ==========================================================

channel_analysis[
    [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumDealsPurchases"
    ]
].plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Shopping Channel Comparison")
plt.xlabel("Cluster")
plt.ylabel("Average Purchases")

plt.tight_layout()
plt.savefig("figures/shopping_channel_comparison.png")
plt.close()

# ==========================================================
# Website Visits
# ==========================================================

website_visits = (
    original_df
    .groupby("Cluster")["NumWebVisitsMonth"]
    .mean()
    .round(2)
)

print("\nWebsite Visits")
print(website_visits)

plt.figure(figsize=(8,5))
plt.bar(
    website_visits.index.astype(str),
    website_visits.values
)

plt.title("Average Website Visits")
plt.xlabel("Cluster")
plt.ylabel("Visits Per Month")

plt.tight_layout()
plt.savefig("figures/website_visits.png")
plt.close()

# ==========================================================
# Recency Comparison
# ==========================================================

recency = (
    original_df
    .groupby("Cluster")["Recency"]
    .mean()
    .round(2)
)

print("\nAverage Recency")
print(recency)

plt.figure(figsize=(8,5))
plt.bar(
    recency.index.astype(str),
    recency.values
)

plt.title("Average Recency by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Days Since Last Purchase")

plt.tight_layout()
plt.savefig("figures/recency_comparison.png")
plt.close()

# ==========================================================
# Customer Segment Identification
# ==========================================================

digital_cluster = channel_analysis["NumWebPurchases"].idxmax()

store_cluster = channel_analysis["NumStorePurchases"].idxmax()

catalog_cluster = channel_analysis["NumCatalogPurchases"].idxmax()

deal_cluster = channel_analysis["NumDealsPurchases"].idxmax()

active_cluster = recency.idxmin()

inactive_cluster = recency.idxmax()

print("\nCustomer Segment Identification")
print("-" * 40)
print(f"Digital-First Customers     : Cluster {digital_cluster}")
print(f"Store-Oriented Customers    : Cluster {store_cluster}")
print(f"Catalog-Oriented Customers  : Cluster {catalog_cluster}")
print(f"Deal-Seeking Customers      : Cluster {deal_cluster}")
print(f"Most Active Customers       : Cluster {active_cluster}")
print(f"Most Inactive Customers     : Cluster {inactive_cluster}")

# ==========================================================
# Engagement Summary
# ==========================================================

engagement_summary = pd.DataFrame({

    "Web Purchases": channel_analysis["NumWebPurchases"],

    "Catalog Purchases": channel_analysis["NumCatalogPurchases"],

    "Store Purchases": channel_analysis["NumStorePurchases"],

    "Deal Purchases": channel_analysis["NumDealsPurchases"],

    "Website Visits": channel_analysis["NumWebVisitsMonth"],

    "Recency": channel_analysis["Recency"]

})

print("\nCustomer Engagement Summary")
print(engagement_summary)

engagement_summary.to_excel(
    "reports/customer_engagement_summary.xlsx"
)

print("\nActivity 3 Completed Successfully")
# ==========================================================
# Activity 4 : Marketing Campaign Analysis
# ==========================================================

print("\n" + "=" * 60)
print("ACTIVITY 4 : MARKETING CAMPAIGN ANALYSIS")
print("=" * 60)

# ==========================================================
# Campaign Columns
# ==========================================================

campaign_columns = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
    "Complain"
]

campaign_analysis = (
    original_df
    .groupby("Cluster")[campaign_columns]
    .mean()
    .round(3)
)

print("\nCampaign Analysis")
print(campaign_analysis)

campaign_analysis.to_excel(
    "reports/campaign_analysis.xlsx"
)

# ==========================================================
# Campaign Response Chart
# ==========================================================

campaign_analysis[
    [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "Response"
    ]
].plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Campaign Response Comparison")
plt.xlabel("Cluster")
plt.ylabel("Average Acceptance Rate")

plt.tight_layout()
plt.savefig(
    "figures/campaign_response_comparison.png"
)

plt.close()

# ==========================================================
# Complaint Comparison
# ==========================================================

complaint = (
    original_df
    .groupby("Cluster")["Complain"]
    .mean()
    .round(3)
)

print("\nComplaint Rate")
print(complaint)

plt.figure(figsize=(8,5))
plt.bar(
    complaint.index.astype(str),
    complaint.values
)

plt.title("Complaint Rate by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Complaint Rate")

plt.tight_layout()
plt.savefig(
    "figures/complaint_rate.png"
)

plt.close()

# ==========================================================
# Campaign Responsive Cluster
# ==========================================================

campaign_score = campaign_analysis[
    [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "Response"
    ]
].sum(axis=1)

responsive_cluster = campaign_score.idxmax()

resistant_cluster = campaign_score.idxmin()

reengagement_cluster = (
    campaign_analysis["Response"]
).idxmin()

# ==========================================================
# Print Results
# ==========================================================

print("\nMarketing Insights")
print("-" * 40)

print(f"Campaign Responsive Customers : Cluster {responsive_cluster}")

print(f"Marketing Resistant Customers : Cluster {resistant_cluster}")

print(f"Customers Requiring Re-engagement : Cluster {reengagement_cluster}")

# ==========================================================
# Marketing Summary
# ==========================================================

marketing_summary = pd.DataFrame({

    "Campaign Score": campaign_score,

    "Final Response":
        campaign_analysis["Response"],

    "Complaint Rate":
        campaign_analysis["Complain"]

})

print("\nMarketing Summary")
print(marketing_summary)

marketing_summary.to_excel(
    "reports/marketing_summary.xlsx"
)

print("\nActivity 4 Completed Successfully")

# ==========================================================
# Activity 5 : Business Segmentation & Cluster Naming
# ==========================================================

print("\n" + "=" * 60)
print("ACTIVITY 5 : BUSINESS SEGMENTATION & CLUSTER NAMING")
print("=" * 60)

# ==========================================================
# Cluster Profile
# ==========================================================

cluster_profile = original_df.groupby("Cluster").agg({

    "Income":"mean",

    "Total_Spending":"mean",

    "Customer_Age":"mean",

    "Family_Size":"mean",

    "Purchase_Frequency":"mean",

    "NumWebPurchases":"mean",

    "NumStorePurchases":"mean",

    "NumCatalogPurchases":"mean",

    "Accepted_Campaigns":"mean",

    "Recency":"mean"

}).round(2)

print("\nCluster Profile")
print(cluster_profile)

# ==========================================================
# Business Segment Naming
# ==========================================================

segment_names = {}

business_recommendations = {}

for cluster in cluster_profile.index:

    spending = cluster_profile.loc[cluster, "Total_Spending"]

    income = cluster_profile.loc[cluster, "Income"]

    web = cluster_profile.loc[cluster, "NumWebPurchases"]

    store = cluster_profile.loc[cluster, "NumStorePurchases"]

    catalog = cluster_profile.loc[cluster, "NumCatalogPurchases"]

    campaigns = cluster_profile.loc[cluster, "Accepted_Campaigns"]

    recency = cluster_profile.loc[cluster, "Recency"]

    # --------------------------
    # Business Rules
    # --------------------------

    # Relative comparison against the OTHER cluster(s), not fixed
    # absolute cutoffs, so segments differentiate even when every
    # cluster happens to be store-dominant.
    spend_rank = cluster_profile["Total_Spending"].rank(ascending=False).loc[cluster]
    income_rank = cluster_profile["Income"].rank(ascending=False).loc[cluster]
    is_channel_web = web >= store and web >= catalog
    is_channel_store = store >= web and store >= catalog
    is_top_half_value = spend_rank <= len(cluster_profile) / 2 and income_rank <= len(cluster_profile) / 2

    if spending >= 1000 and income >= 60000:

        name = "Premium Loyal Customers"

        strategy = "Offer VIP membership, premium rewards and exclusive products."

    elif is_channel_web and is_top_half_value:

        name = "Digital-First High-Value Buyers"

        strategy = "Focus on email campaigns, mobile app offers and online discounts."

    elif is_channel_web:

        name = "Digital-First Budget Buyers"

        strategy = "Focus on low-cost email/app promotions and value bundles."

    elif is_channel_store and is_top_half_value:

        name = "High-Value Store-Oriented Customers"

        strategy = "Increase in-store promotions, loyalty rewards and cross-sell at checkout."

    elif is_channel_store and campaigns >= 2:

        name = "Campaign-Responsive Store Shoppers"

        strategy = "Continue personalized in-store and campaign-based offers."

    elif is_channel_store and spending < 300:

        name = "Budget-Conscious Store Shoppers"

        strategy = "Provide discount coupons, deals and re-engagement offers."

    elif is_channel_store:

        name = "Store-Oriented Traditional Customers"

        strategy = "Maintain in-store engagement with periodic loyalty incentives."

    elif campaigns >= 2:

        name = "Campaign-Responsive Customers"

        strategy = "Continue personalized marketing campaigns."

    elif spending < 300:

        name = "Low-Value Inactive Customers"

        strategy = "Provide discount coupons and re-engagement offers."

    else:

        name = "Regular Customers"

        strategy = "Maintain engagement with personalized recommendations."

    segment_names[cluster] = name

    business_recommendations[cluster] = strategy

# ==========================================================
# Create Business Segment Table
# ==========================================================

business_segments = pd.DataFrame({

    "Business Segment":

        pd.Series(segment_names),

    "Marketing Strategy":

        pd.Series(business_recommendations)

})

print("\nBusiness Segments")
print(business_segments)

# ==========================================================
# Save Reports
# ==========================================================

business_segments.to_excel(
    "reports/business_segment_names.xlsx"
)

cluster_profile.to_excel(
    "reports/cluster_profile.xlsx"
)

# ==========================================================
# Add Business Segment to Dataset
# ==========================================================

original_df["Business_Segment"] = (
    original_df["Cluster"].map(segment_names)
)

original_df.to_excel(
    "reports/customer_profile_dataset.xlsx",
    index=False
)

print("\nBusiness Segment Names Saved Successfully")

print("\nActivity 5 Completed Successfully")

# ==========================================================
# Activity 6 : Customer Persona Development
# ==========================================================

print("\n" + "=" * 60)
print("ACTIVITY 6 : CUSTOMER PERSONA DEVELOPMENT")
print("=" * 60)

# ==========================================================
# Create Customer Personas
# ==========================================================

persona_list = []

for cluster in cluster_profile.index:

    profile = cluster_profile.loc[cluster]

    # ----------------------------
    # Age Group
    # ----------------------------

    age = profile["Customer_Age"]

    if age < 35:
        age_group = "Young Adults"

    elif age < 50:
        age_group = "Middle Aged"

    else:
        age_group = "Senior Adults"

    # ----------------------------
    # Income Level
    # ----------------------------

    income = profile["Income"]

    if income >= 70000:
        income_level = "High"

    elif income >= 40000:
        income_level = "Medium"

    else:
        income_level = "Low"

    # ----------------------------
    # Family Status
    # ----------------------------

    if profile["Family_Size"] <= 2:
        family_status = "Small Family"

    elif profile["Family_Size"] <= 4:
        family_status = "Medium Family"

    else:
        family_status = "Large Family"

    # ----------------------------
    # Preferred Shopping Channel
    # ----------------------------

    channels = {
        "Web": profile["NumWebPurchases"],
        "Store": profile["NumStorePurchases"],
        "Catalog": profile["NumCatalogPurchases"]
    }

    preferred_channel = max(channels, key=channels.get)

    # ----------------------------
    # Marketing Response
    # ----------------------------

    if profile["Accepted_Campaigns"] >= 2:
        marketing_response = "Highly Responsive"

    elif profile["Accepted_Campaigns"] >= 1:
        marketing_response = "Moderately Responsive"

    else:
        marketing_response = "Low Response"

    # ----------------------------
    # Customer Lifetime Value
    # ----------------------------

    if profile["Total_Spending"] >= 1000:
        clv = "High"

    elif profile["Total_Spending"] >= 500:
        clv = "Medium"

    else:
        clv = "Low"

    # ----------------------------
    # Persona Name
    # ----------------------------

    persona_name = f"Cluster {cluster} Persona"

    # ----------------------------
    # Customer Challenge
    # ----------------------------

    if clv == "High":
        challenge = "Maintain long-term loyalty."

    elif clv == "Medium":
        challenge = "Increase purchase frequency."

    else:
        challenge = "Re-engage inactive customers."

    # ----------------------------
    # Marketing Strategy
    # ----------------------------

    strategy = business_recommendations[cluster]

    # ----------------------------
    # Store Persona
    # ----------------------------

    persona_list.append({

        "Cluster": cluster,

        "Persona Name": persona_name,

        "Age Group": age_group,

        "Income Level": income_level,

        "Family Status": family_status,

        "Preferred Channel": preferred_channel,

        "Marketing Response": marketing_response,

        "Customer Lifetime Value": clv,

        "Customer Challenge": challenge,

        "Marketing Strategy": strategy

    })

# ==========================================================
# Persona DataFrame
# ==========================================================

persona_df = pd.DataFrame(persona_list)

print("\nCustomer Personas")
print(persona_df)

# ==========================================================
# Save Persona Report
# ==========================================================

persona_df.to_excel(
    "reports/customer_personas.xlsx",
    index=False
)

print("\nCustomer Personas Saved Successfully")

print("\nActivity 6 Completed Successfully")
# ==========================================================
# Activity 7 : Cluster Visualization & Final Documentation
# ==========================================================

print("\n" + "=" * 60)
print("ACTIVITY 7 : CLUSTER VISUALIZATION & FINAL DOCUMENTATION")
print("=" * 60)

# ==========================================================
# Cluster Distribution Chart
# ==========================================================

cluster_distribution = original_df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(cluster_distribution.index.astype(str),
        cluster_distribution.values)

plt.title("Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("figures/cluster_distribution.png")
plt.close()

# ==========================================================
# Income Comparison Chart
# ==========================================================

income_chart = original_df.groupby("Cluster")["Income"].mean()

plt.figure(figsize=(8,5))
plt.bar(income_chart.index.astype(str),
        income_chart.values)

plt.title("Average Income by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Income")

plt.tight_layout()
plt.savefig("figures/income_comparison_chart.png")
plt.close()

# ==========================================================
# Spending Comparison Chart
# ==========================================================

spending_chart = original_df.groupby("Cluster")["Total_Spending"].mean()

plt.figure(figsize=(8,5))
plt.bar(spending_chart.index.astype(str),
        spending_chart.values)

plt.title("Average Spending by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Total Spending")

plt.tight_layout()
plt.savefig("figures/spending_comparison_chart.png")
plt.close()

# ==========================================================
# Product Preference Heatmap
# ==========================================================

product_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

heatmap_data = (
    original_df
    .groupby("Cluster")[product_columns]
    .mean()
)

plt.figure(figsize=(10,6))

plt.imshow(
    heatmap_data,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(product_columns)),
    product_columns,
    rotation=45
)

plt.yticks(
    range(len(heatmap_data.index)),
    heatmap_data.index
)

plt.title("Product Preference Heatmap")

plt.tight_layout()

plt.savefig("figures/product_heatmap.png")

plt.close()

# ==========================================================
# Channel Comparison Chart
# ==========================================================

channel_chart = original_df.groupby("Cluster")[

    [
        "NumWebPurchases",
        "NumStorePurchases",
        "NumCatalogPurchases"
    ]

].mean()

channel_chart.plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Purchasing Channel Comparison")
plt.xlabel("Cluster")
plt.ylabel("Average Purchases")

plt.tight_layout()

plt.savefig(
    "figures/channel_comparison.png"
)

plt.close()

# ==========================================================
# Campaign Response Chart
# ==========================================================

campaign_chart = original_df.groupby("Cluster")[
    "Response"
].mean()

plt.figure(figsize=(8,5))

plt.bar(
    campaign_chart.index.astype(str),
    campaign_chart.values
)

plt.title("Campaign Response Comparison")
plt.xlabel("Cluster")
plt.ylabel("Response Rate")

plt.tight_layout()

plt.savefig(
    "figures/campaign_response_chart.png"
)

plt.close()

# ==========================================================
# Recency Comparison
# ==========================================================

recency_chart = original_df.groupby("Cluster")[
    "Recency"
].mean()

plt.figure(figsize=(8,5))

plt.bar(
    recency_chart.index.astype(str),
    recency_chart.values
)

plt.title("Average Recency")
plt.xlabel("Cluster")
plt.ylabel("Days")

plt.tight_layout()

plt.savefig(
    "figures/recency_chart.png"
)

plt.close()

# ==========================================================
# Segment Comparison Table
# ==========================================================

segment_table = cluster_profile.copy()

segment_table["Business Segment"] = (
    segment_table.index.map(segment_names)
)

segment_table.to_excel(
    "reports/segment_comparison_table.xlsx"
)

# ==========================================================
# Final Customer Profiling Report
# ==========================================================

with open(
    "reports/final_customer_profiling_report.txt",
    "w",
    encoding="utf-8"
) as report:

    report.write("MODULE 7\n")
    report.write("Customer Profiling Report\n\n")

    report.write("Business Segments\n")
    report.write("==============================\n\n")

    for cluster in segment_names:

        report.write(
            f"Cluster {cluster}\n"
        )

        report.write(
            f"Segment : {segment_names[cluster]}\n"
        )

        report.write(
            f"Strategy : {business_recommendations[cluster]}\n\n"
        )

print("\nFinal Customer Profiling Report Saved")

print("\nActivity 7 Completed Successfully")

print("\n" + "=" * 60)
print("MODULE 7 COMPLETED SUCCESSFULLY")
print("=" * 60)