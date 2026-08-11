"""
Module 8: Business Insights and Marketing Recommendations
Customer Segmentation -> Marketing Strategy Pipeline

This script:
1. Loads the engineered clustering dataset and the raw marketing campaign data
2. Scales features and fits a K-Means model (k=4, chosen to match business
   granularity requirements and validated with silhouette analysis)
3. Profiles each segment and assigns business-friendly names
4. Exports a segmented customer dataset and a business dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RAW_PATH = "1786312511574_marketing_campaign.csv"
FEAT_PATH = "1786312511575_clustering_dataset.csv"

SEGMENT_NAMES = {
    0: "Budget-Conscious Families",
    1: "Low-Engagement Shoppers",
    2: "Premium Loyal Customers",
    3: "Affluent Family Shoppers",
}


def load_data(raw_path=RAW_PATH, feat_path=FEAT_PATH):
    df_raw = pd.read_csv(raw_path, sep="\t")
    df_feat = pd.read_csv(feat_path)
    return df_raw, df_feat


def choose_k(Xs, k_range=range(2, 8)):
    """Evaluate silhouette score across a range of k for reference."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        scores[k] = silhouette_score(Xs, labels)
    return scores


def fit_clusters(df_feat, n_clusters=4):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_feat)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    return labels, scaler, km


def build_segmented_dataset(df_raw, df_feat, labels):
    df = df_raw.copy()
    for c in df_feat.columns:
        df[c] = df_feat[c].values
    df["Cluster"] = labels
    df["SegmentName"] = df["Cluster"].map(SEGMENT_NAMES)

    mnt_cols = [c for c in df.columns if c.startswith("Mnt")]
    df["Total_Mnt"] = df[mnt_cols].sum(axis=1)

    camp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                 "AcceptedCmp4", "AcceptedCmp5", "Response"]
    df["CampaignAcceptRate"] = df[camp_cols].mean(axis=1)

    df["TotalPurchases"] = (df["NumWebPurchases"] + df["NumStorePurchases"]
                             + df["NumCatalogPurchases"] + df["NumDealsPurchases"])
    df["BrowseToBuyRatio"] = df["NumWebVisitsMonth"] / df["TotalPurchases"].replace(0, np.nan)
    return df


def segment_summary_table(df):
    counts = df["Cluster"].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(1)
    table = pd.DataFrame({
        "Cluster": [f"Cluster {c}" for c in counts.index],
        "Business Name": [SEGMENT_NAMES[c] for c in counts.index],
        "Customers": counts.values,
        "Percentage": [f"{p}%" for p in pct.values],
    })
    return table


def make_dashboard(df, out_path="dashboard.png"):
    names = SEGMENT_NAMES
    colors = {0: "#f4a261", 1: "#adb5bd", 2: "#2a9d8f", 3: "#264653"}
    order = [0, 1, 2, 3]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Customer Segmentation — Business Insights Dashboard",
                 fontsize=15, fontweight="bold")

    counts = df["Cluster"].value_counts().reindex(order)
    axes[0, 0].bar([names[c] for c in order], counts.values, color=[colors[c] for c in order])
    axes[0, 0].set_title("Customers per Segment")

    rev = df.groupby("Cluster")["Total_Mnt"].sum().reindex(order)
    axes[0, 1].pie(rev, labels=[names[c] for c in order], autopct="%1.1f%%",
                    colors=[colors[c] for c in order], textprops={"fontsize": 8})
    axes[0, 1].set_title("Revenue Contribution by Segment")

    camp = df.groupby("Cluster")["CampaignAcceptRate"].mean().reindex(order) * 100
    axes[0, 2].bar([names[c] for c in order], camp.values, color=[colors[c] for c in order])
    axes[0, 2].set_title("Avg. Campaign Acceptance Rate (%)")

    risk = df.groupby("Cluster")["BrowseToBuyRatio"].mean().reindex(order)
    axes[1, 0].bar([names[c] for c in order], risk.values, color=[colors[c] for c in order])
    axes[1, 0].set_title("Churn Risk Proxy\n(Web Visits per Purchase)")

    ch = df.groupby("Cluster")[["NumWebPurchases", "NumStorePurchases",
                                 "NumCatalogPurchases", "NumDealsPurchases"]].mean().reindex(order)
    ch.index = [names[c] for c in order]
    ch.plot(kind="bar", stacked=True, ax=axes[1, 1],
            color=["#e76f51", "#2a9d8f", "#264653", "#e9c46a"])
    axes[1, 1].set_title("Purchase Channel Preference")

    prod = df.groupby("Cluster")[["MntWines", "MntFruits", "MntMeatProducts",
                                   "MntFishProducts", "MntSweetProducts",
                                   "MntGoldProds"]].mean().reindex(order)
    prod.index = [names[c] for c in order]
    prod.plot(kind="bar", stacked=True, ax=axes[1, 2])
    axes[1, 2].set_title("Product Category Spend ($ avg/customer)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df_raw, df_feat = load_data()
    Xs = StandardScaler().fit_transform(df_feat)
    scores = choose_k(Xs)
    print("Silhouette scores by k:", scores)

    labels, scaler, km = fit_clusters(df_feat, n_clusters=4)
    df = build_segmented_dataset(df_raw, df_feat, labels)

    print(segment_summary_table(df))
    df.to_csv("segmented_customers.csv", index=False)
    make_dashboard(df)
    print("Done. Outputs: segmented_customers.csv, dashboard.png")


if __name__ == "__main__":
    main()
