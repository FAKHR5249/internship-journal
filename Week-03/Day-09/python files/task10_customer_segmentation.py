# ==========================================================
# AI Lab 99 Internship Program 2026
# Module 4 - Exploratory Data Analysis (EDA)
# Task 10 - Customer Segmentation
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

# Total Spending

spending_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

df["Total_Spending"] = df[spending_columns].sum(axis=1)


# Total Purchases

purchase_columns = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]

df["Total_Purchases"] = df[purchase_columns].sum(axis=1)


# Total Campaign Response

campaign_columns = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5"
]

df["Campaign_Response"] = df[campaign_columns].sum(axis=1)


# ----------------------------------------------------------
# Customer Segmentation
# ----------------------------------------------------------

# High Value Customers
# Spending above 75th percentile

high_value_limit = df["Total_Spending"].quantile(0.75)

df["Customer_Value"] = df["Total_Spending"].apply(
    lambda x: "High Value" if x >= high_value_limit else "Low Value"
)


# Frequent Buyers

purchase_limit = df["Total_Purchases"].quantile(0.75)

df["Purchase_Frequency"] = df["Total_Purchases"].apply(
    lambda x: "Frequent Buyer" if x >= purchase_limit else "Normal Buyer"
)


# Discount Seekers

df["Discount_Behavior"] = df["NumDealsPurchases"].apply(
    lambda x: "Discount Seeker" if x >= 3 else "Regular Buyer"
)


# Campaign Responders

df["Campaign_Status"] = df["Campaign_Response"].apply(
    lambda x: "Campaign Responder" if x > 0 else "No Response"
)


# Loyal Customers

df["Loyalty_Status"] = (
    (df["Total_Purchases"] >= purchase_limit) &
    (df["Total_Spending"] >= high_value_limit)
)

df["Loyalty_Status"] = df["Loyalty_Status"].apply(
    lambda x: "Loyal Customer" if x else "Other Customer"
)


# Inactive Customers

df["Activity_Status"] = df["Recency"].apply(
    lambda x: "Inactive Customer" if x > 60
    else "Active Customer"
)


# ----------------------------------------------------------
# Segmentation Results
# ----------------------------------------------------------

print("="*70)
print("CUSTOMER SEGMENTATION ANALYSIS")
print("="*70)


print("\nCustomer Value Segmentation")
print("-"*50)

print(df["Customer_Value"].value_counts())


print("\nPurchase Frequency Segmentation")
print("-"*50)

print(df["Purchase_Frequency"].value_counts())


print("\nDiscount Behavior Segmentation")
print("-"*50)

print(df["Discount_Behavior"].value_counts())


print("\nCampaign Response Segmentation")
print("-"*50)

print(df["Campaign_Status"].value_counts())


print("\nLoyal Customer Segmentation")
print("-"*50)

print(df["Loyalty_Status"].value_counts())


print("\nCustomer Activity Segmentation")
print("-"*50)

print(df["Activity_Status"].value_counts())


# ----------------------------------------------------------
# Top High Value Customers
# ----------------------------------------------------------

print("\n")
print("="*70)
print("TOP 10 HIGH VALUE CUSTOMERS")
print("="*70)


print(
    df.sort_values(
        "Total_Spending",
        ascending=False
    )[["ID","Total_Spending"]].head(10)
)


# ----------------------------------------------------------
# Visualizations
# ----------------------------------------------------------

sns.set_style("whitegrid")


# Customer Value Chart

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Customer_Value"
)

plt.title("High Value vs Low Value Customers")
plt.xlabel("Customer Segment")
plt.ylabel("Customers")

plt.show()



# Purchase Frequency Chart

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Purchase_Frequency"
)

plt.title("Customer Purchase Frequency")
plt.xlabel("Segment")
plt.ylabel("Customers")

plt.show()



# Campaign Response Chart

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Campaign_Status"
)

plt.title("Campaign Responders")
plt.xlabel("Status")
plt.ylabel("Customers")

plt.show()



# Spending Distribution

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Customer_Value",
    y="Total_Spending"
)

plt.title("Spending Distribution by Customer Value")

plt.show()


# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\n")
print("="*70)
print("BUSINESS SEGMENTATION SUMMARY")
print("="*70)


print(
    "High Value Customers:",
    len(df[df["Customer_Value"]=="High Value"])
)


print(
    "Frequent Buyers:",
    len(df[df["Purchase_Frequency"]=="Frequent Buyer"])
)


print(
    "Discount Seekers:",
    len(df[df["Discount_Behavior"]=="Discount Seeker"])
)


print(
    "Campaign Responders:",
    len(df[df["Campaign_Status"]=="Campaign Responder"])
)


print(
    "Loyal Customers:",
    len(df[df["Loyalty_Status"]=="Loyal Customer"])
)


print(
    "Inactive Customers:",
    len(df[df["Activity_Status"]=="Inactive Customer"])
)


print("\nTask 10 Completed Successfully.")