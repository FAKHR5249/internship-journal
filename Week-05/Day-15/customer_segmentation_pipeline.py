# ==========================================================
# Customer Segmentation Pipeline
# Module 6 - Customer Personality Analysis
# Author: Fakhr Ul Islam
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import time
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

# ==========================================================
# Create Output Folders
# ==========================================================

folders = [
    "data",
    "data/processed",
    "models",
    "reports",
    "plots",
    "pipeline"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("="*60)
print("Customer Segmentation Pipeline Started")
print("="*60)

# ==========================================================
# Task 1
# Dataset Validation
# ==========================================================

print("\nLoading Dataset...")

dataset_path = "marketing_campaign.csv"

df = pd.read_csv(dataset_path, sep="\t")

print("Dataset Loaded Successfully")

print("\nShape")
print(df.shape)

print("\nDataset Info")
df.info()

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

print("\nData Types")
print(df.dtypes)

print("\nSummary Statistics")
print(df.describe().T)

print("\nColumn Names")

for col in df.columns:
    print(col)

# ==========================================================
# Remove Unnecessary Columns
# (Dt_Customer is kept for now - needed for Customer_Tenure
#  feature engineering below, then dropped afterwards)
# ==========================================================

drop_columns = []

if "ID" in df.columns:
    drop_columns.append("ID")

if len(drop_columns) > 0:
    df.drop(columns=drop_columns, inplace=True)

print("\nRemoved Columns")
print(drop_columns)

print("\nRemaining Shape")
print(df.shape)

# ==========================================================
# Validation Report
# ==========================================================

report = pd.DataFrame({
    "Rows":[df.shape[0]],
    "Columns":[df.shape[1]],
    "Missing Values":[df.isnull().sum().sum()],
    "Duplicate Records":[df.duplicated().sum()]
})

report.to_csv(
    "reports/dataset_validation_report.csv",
    index=False
)

print("\nDataset Validation Completed")
print(report)

# ==========================================================
# Missing Value Handling
# ==========================================================

print("\nHandling Missing Values...")

# Fill Income missing values with median
df["Income"] = df["Income"].fillna(df["Income"].median())

print("Remaining Missing Values")
print(df.isnull().sum())

# ==========================================================
# Feature Engineering
# ==========================================================

print("\n" + "="*60)
print("Feature Engineering")
print("="*60)

current_year = 2026

# Customer Age
df["Customer_Age"] = current_year - df["Year_Birth"]

# Total Spending
df["Total_Spending"] = (
    df["MntWines"] +
    df["MntFruits"] +
    df["MntMeatProducts"] +
    df["MntFishProducts"] +
    df["MntSweetProducts"] +
    df["MntGoldProds"]
)

# Total Children
df["Total_Children"] = (
    df["Kidhome"] +
    df["Teenhome"]
)

# Family Size
df["Family_Size"] = df["Total_Children"] + 2

# Purchase Frequency
df["Purchase_Frequency"] = (
    df["NumWebPurchases"] +
    df["NumCatalogPurchases"] +
    df["NumStorePurchases"]
)

# Accepted Campaigns
df["Accepted_Campaigns"] = (
    df["AcceptedCmp1"] +
    df["AcceptedCmp2"] +
    df["AcceptedCmp3"] +
    df["AcceptedCmp4"] +
    df["AcceptedCmp5"] +
    df["Response"]
)

# Customer Tenure
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True
)

reference_date = pd.Timestamp("2026-01-01")

df["Customer_Tenure"] = (
    reference_date - df["Dt_Customer"]
).dt.days

# Dt_Customer no longer needed after Customer_Tenure is derived
df.drop(columns=["Dt_Customer"], inplace=True)

print("\nDt_Customer Column Removed (used to derive Customer_Tenure)")

print("\nNew Features Created")

new_features = [
    "Customer_Age",
    "Total_Spending",
    "Family_Size",
    "Total_Children",
    "Purchase_Frequency",
    "Accepted_Campaigns",
    "Customer_Tenure"
]

print(df[new_features].head())

# ==========================================================
# Task 2
# Feature Selection for Clustering
# ==========================================================

print("\n" + "="*60)
print("Task 2 : Feature Selection")
print("="*60)

# Recommended Features
selected_features = [
    "Customer_Age",
    "Income",
    "Total_Spending",
    "Recency",
    "Customer_Tenure",
    "Family_Size",
    "Total_Children",
    "Purchase_Frequency",
    "Accepted_Campaigns",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumCatalogPurchases"
]

# Check if features exist
available_features = []
missing_features = []

for feature in selected_features:
    if feature in df.columns:
        available_features.append(feature)
    else:
        missing_features.append(feature)

print("\nAvailable Features")
print(available_features)

if len(missing_features) > 0:
    print("\nMissing Features")
    print(missing_features)

cluster_df = df[available_features].copy()

print("\nSelected Dataset Shape")
print(cluster_df.shape)

# ==========================================================
# Correlation Matrix
# ==========================================================

corr = cluster_df.corr()

plt.figure(figsize=(12,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "plots/correlation_heatmap.png",
    dpi=300
)

plt.close()

# ==========================================================
# Highly Correlated Features
# ==========================================================

print("\nHighly Correlated Features (>0.90)\n")

threshold = 0.90

high_corr = []

for i in range(len(corr.columns)):
    for j in range(i):

        value = abs(corr.iloc[i, j])

        if value > threshold:

            feature1 = corr.columns[i]
            feature2 = corr.columns[j]

            high_corr.append(
                [feature1, feature2, value]
            )

if len(high_corr) == 0:
    print("No Highly Correlated Features Found")

else:

    corr_df = pd.DataFrame(
        high_corr,
        columns=[
            "Feature1",
            "Feature2",
            "Correlation"
        ]
    )

    print(corr_df)

    corr_df.to_csv(
        "reports/high_correlation_report.csv",
        index=False
    )

# ==========================================================
# Save Selected Dataset
# ==========================================================

cluster_df.to_csv(
    "data/processed/clustering_dataset.csv",
    index=False
)

print("\nSelected Dataset Saved")

# ==========================================================
# Task 3
# Feature Scaling
# ==========================================================

print("\n" + "="*60)
print("Task 3 : Feature Scaling")
print("="*60)

# -----------------------------
# StandardScaler
# -----------------------------

standard_scaler = StandardScaler()

standard_scaled = standard_scaler.fit_transform(cluster_df)

standard_scaled_df = pd.DataFrame(
    standard_scaled,
    columns=cluster_df.columns
)

standard_scaled_df.to_csv(
    "data/processed/standard_scaled_dataset.csv",
    index=False
)

print("Standard Scaling Completed")

# -----------------------------
# MinMaxScaler
# -----------------------------

minmax_scaler = MinMaxScaler()

minmax_scaled = minmax_scaler.fit_transform(cluster_df)

minmax_scaled_df = pd.DataFrame(
    minmax_scaled,
    columns=cluster_df.columns
)

minmax_scaled_df.to_csv(
    "data/processed/minmax_scaled_dataset.csv",
    index=False
)

print("MinMax Scaling Completed")

# ==========================================================
# Save Scalers
# ==========================================================

joblib.dump(
    standard_scaler,
    "models/standard_scaler.pkl"
)

joblib.dump(
    minmax_scaler,
    "models/minmax_scaler.pkl"
)

print("\nScalers Saved Successfully")

# ==========================================================
# Scaling Comparison
# ==========================================================

comparison = pd.DataFrame({
    "Feature": cluster_df.columns,
    "Original Mean": cluster_df.mean().values,
    "Original Std": cluster_df.std().values,
    "Standard Mean": standard_scaled_df.mean().values,
    "Standard Std": standard_scaled_df.std().values,
    "MinMax Min": minmax_scaled_df.min().values,
    "MinMax Max": minmax_scaled_df.max().values
})

comparison.to_csv(
    "reports/scaling_comparison.csv",
    index=False
)

print("\nScaling Comparison Saved")

# ==========================================================
# Save Preprocessing Pipeline
# ==========================================================

pipeline = {
    "selected_features": available_features,
    "standard_scaler": standard_scaler,
    "minmax_scaler": minmax_scaler
}

joblib.dump(
    pipeline,
    "pipeline/preprocessing_pipeline.pkl"
)

print("\nPreprocessing Pipeline Saved")

print("\nTask 2 & Task 3 Completed Successfully")


# ==========================================================
# Task 4
# Exploratory Cluster Analysis (EDA)
# ==========================================================

print("\n" + "=" * 60)
print("Task 4 : Exploratory Cluster Analysis")
print("=" * 60)

eda_df = cluster_df.copy()

# ==========================================================
# Basic Statistics
# ==========================================================

print("\nBasic Statistics")
print(eda_df.describe())

eda_df.describe().T.to_csv(
    "reports/eda_summary_statistics.csv"
)

# ==========================================================
# Feature Distributions
# ==========================================================

print("\nGenerating Feature Distribution Plots...")

for column in eda_df.columns:

    plt.figure(figsize=(7,5))

    sns.histplot(
        eda_df[column],
        kde=True,
        bins=30
    )

    plt.title(f"{column} Distribution")

    plt.tight_layout()

    plt.savefig(
        f"plots/distribution_{column}.png",
        dpi=300
    )

    plt.close()

print("Distribution Plots Saved")

# ==========================================================
# Boxplots
# ==========================================================

print("\nGenerating Boxplots...")

for column in eda_df.columns:

    plt.figure(figsize=(6,4))

    sns.boxplot(
        x=eda_df[column]
    )

    plt.title(f"{column} Boxplot")

    plt.tight_layout()

    plt.savefig(
        f"plots/boxplot_{column}.png",
        dpi=300
    )

    plt.close()

print("Boxplots Saved")

# ==========================================================
# Pair Plot
# ==========================================================

print("\nGenerating Pair Plot...")

pair_columns = eda_df.columns[:5]

pairplot = sns.pairplot(
    eda_df[pair_columns]
)

pairplot.savefig(
    "plots/pairplot.png",
    dpi=300
)

plt.close()

print("Pair Plot Saved")

# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    eda_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "plots/correlation_heatmap_eda.png",
    dpi=300
)

plt.close()

print("Correlation Heatmap Saved")

# ==========================================================
# PCA Visualization
# ==========================================================

print("\nRunning PCA...")

pca = PCA(n_components=2)

pca_data = pca.fit_transform(standard_scaled_df)

pca_df = pd.DataFrame(
    pca_data,
    columns=["PC1","PC2"]
)

plt.figure(figsize=(8,6))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    alpha=0.6
)

plt.title("PCA Visualization")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()

plt.savefig(
    "plots/pca_visualization.png",
    dpi=300
)

plt.close()

print("PCA Visualization Saved")

# ==========================================================
# PCA Variance
# ==========================================================

variance = pd.DataFrame({

    "Component":[
        "PC1",
        "PC2"
    ],

    "Explained Variance":[
        pca.explained_variance_ratio_[0],
        pca.explained_variance_ratio_[1]
    ]

})

variance.to_csv(
    "reports/pca_variance.csv",
    index=False
)

print("\nExplained Variance")
print(variance)

# ==========================================================
# Outlier Detection (IQR)
# ==========================================================

print("\nDetecting Outliers...")

outlier_summary = []

for column in eda_df.columns:

    Q1 = eda_df[column].quantile(0.25)
    Q3 = eda_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = (
        (eda_df[column] < lower) |
        (eda_df[column] > upper)
    ).sum()

    outlier_summary.append([
        column,
        outliers
    ])

outlier_report = pd.DataFrame(
    outlier_summary,
    columns=[
        "Feature",
        "Outliers"
    ]
)

outlier_report.to_csv(
    "reports/outlier_report.csv",
    index=False
)

print(outlier_report)

# ==========================================================
# Feature Distribution Report
# ==========================================================

distribution_report = pd.DataFrame({

    "Feature": eda_df.columns,

    "Mean": eda_df.mean().values,

    "Median": eda_df.median().values,

    "Std": eda_df.std().values,

    "Minimum": eda_df.min().values,

    "Maximum": eda_df.max().values

})

distribution_report.to_csv(
    "reports/feature_distribution_report.csv",
    index=False
)

print("\nDistribution Report Saved")

print("\nTask 4 Completed Successfully")

# ==========================================================
# Task 5
# Determine Optimal Number of Clusters (Elbow Method)
# ==========================================================

print("\n" + "=" * 60)
print("Task 5 : Determine Optimal Number of Clusters")
print("=" * 60)

# Standard Scaled Dataset
X = standard_scaled_df.copy()

wcss = []
inertia_values = []

k_values = range(2, 11)

print("\nRunning K-Means for K = 2 to 10...\n")

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    model.fit(X)

    wcss.append(model.inertia_)
    inertia_values.append(model.inertia_)

    print(f"K = {k}   WCSS = {model.inertia_:.2f}")

# ==========================================================
# Save WCSS Table
# ==========================================================

elbow_df = pd.DataFrame({
    "K": list(k_values),
    "WCSS": wcss,
    "Inertia": inertia_values
})

elbow_df.to_csv(
    "reports/elbow_results.csv",
    index=False
)

print("\nElbow Results Saved")

# ==========================================================
# Plot Elbow Curve
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(
    list(k_values),
    wcss,
    marker="o",
    linewidth=2
)

plt.xticks(list(k_values))

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "plots/elbow_curve.png",
    dpi=300
)

plt.close()

print("Elbow Curve Saved")

# ==========================================================
# Percentage Reduction
# ==========================================================

print("\nPercentage Reduction in WCSS")

k_values_list = list(k_values)

for i in range(1, len(wcss)):

    reduction = (
        (wcss[i-1] - wcss[i]) /
        wcss[i-1]
    ) * 100

    print(
        f"K={k_values_list[i]}  Reduction = {reduction:.2f}%"
    )

# ==========================================================
# Automatic Elbow Detection
# ==========================================================

reductions = []

for i in range(1, len(wcss)):

    reductions.append(
        (wcss[i-1] - wcss[i]) / wcss[i-1]
    )

best_k = k_values_list[np.argmax(reductions) + 1]

print("\nSuggested K :", best_k)

with open(
    "reports/optimal_k.txt",
    "w"
) as f:

    f.write(f"Suggested Optimal K = {best_k}")

print("Optimal K Report Saved")

print("\nTask 5 Completed Successfully")

# ==========================================================
# Task 6
# Silhouette Score Analysis
# ==========================================================

print("\n" + "=" * 60)
print("Task 6 : Silhouette Score Analysis")
print("=" * 60)

from sklearn.metrics import silhouette_samples

X = standard_scaled_df.copy()

k_range = range(2, 11)

silhouette_scores = []

for k in k_range:

    print(f"\nEvaluating K = {k}")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    silhouette_scores.append(score)

    print(f"Silhouette Score : {score:.4f}")

# ==========================================================
# Save Results
# ==========================================================

silhouette_df = pd.DataFrame({

    "K": list(k_range),

    "Silhouette Score": silhouette_scores

})

silhouette_df.to_csv(

    "reports/silhouette_scores.csv",

    index=False

)

print("\nSilhouette Score Report Saved")

# ==========================================================
# Plot Silhouette Scores
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(

    list(k_range),

    silhouette_scores,

    marker="o",

    linewidth=2

)

plt.xticks(list(k_range))

plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Score")

plt.title("Silhouette Score vs Number of Clusters")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    "plots/silhouette_score_curve.png",

    dpi=300

)

plt.close()

print("Silhouette Curve Saved")

# ==========================================================
# Best K
# ==========================================================

best_index = np.argmax(silhouette_scores)

best_k_silhouette = list(k_range)[best_index]

best_score = silhouette_scores[best_index]

print("\nBest K :", best_k_silhouette)

print("Best Silhouette Score :", round(best_score,4))

# ==========================================================
# Silhouette Plot
# ==========================================================

final_model = KMeans(

    n_clusters=best_k_silhouette,

    random_state=42,

    n_init=10

)

cluster_labels_temp = final_model.fit_predict(X)

sample_scores = silhouette_samples(

    X,

    cluster_labels_temp

)

plt.figure(figsize=(9,6))

y_lower = 10

for i in range(best_k_silhouette):

    values = sample_scores[cluster_labels_temp == i]

    values.sort()

    size = len(values)

    y_upper = y_lower + size

    plt.fill_betweenx(

        np.arange(y_lower, y_upper),

        0,

        values,

        alpha=0.7

    )

    plt.text(

        -0.05,

        y_lower + size / 2,

        str(i)

    )

    y_lower = y_upper + 10

plt.axvline(

    x=best_score,

    color="red",

    linestyle="--"

)

plt.title("Silhouette Plot")

plt.xlabel("Silhouette Coefficient")

plt.ylabel("Clusters")

plt.tight_layout()

plt.savefig(

    "plots/silhouette_plot.png",

    dpi=300

)

plt.close()

print("Silhouette Plot Saved")

print("\nTask 6 Completed Successfully")

# ==========================================================
# Task 7
# Davies-Bouldin Index Analysis
# ==========================================================

print("\n" + "=" * 60)
print("Task 7 : Davies-Bouldin Index Analysis")
print("=" * 60)

X = standard_scaled_df.copy()

k_range = range(2, 11)

db_scores = []

for k in k_range:

    print(f"\nEvaluating K = {k}")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    db = davies_bouldin_score(X, labels)

    db_scores.append(db)

    print(f"Davies-Bouldin Index : {db:.4f}")

# ==========================================================
# Save DB Results
# ==========================================================

db_df = pd.DataFrame({

    "K": list(k_range),

    "Davies_Bouldin_Index": db_scores

})

db_df.to_csv(

    "reports/davies_bouldin_scores.csv",

    index=False

)

print("\nDavies-Bouldin Report Saved")

# ==========================================================
# Plot DB Curve
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(

    list(k_range),

    db_scores,

    marker="o",

    linewidth=2

)

plt.xticks(list(k_range))

plt.xlabel("Number of Clusters")

plt.ylabel("Davies-Bouldin Index")

plt.title("Davies-Bouldin Index vs Number of Clusters")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    "plots/davies_bouldin_curve.png",

    dpi=300

)

plt.close()

print("Davies-Bouldin Curve Saved")

# ==========================================================
# Best K using DB Index
# ==========================================================

best_db_index = np.argmin(db_scores)

best_k_db = list(k_range)[best_db_index]

best_db_score = db_scores[best_db_index]

print("\nBest K (DB Index):", best_k_db)
print("Best DB Index:", round(best_db_score, 4))

# ==========================================================
# Comparison with Silhouette
# ==========================================================

comparison_df = pd.DataFrame({

    "K": list(k_range),

    "Silhouette Score": silhouette_scores,

    "Davies-Bouldin Index": db_scores

})

comparison_df.to_csv(

    "reports/clustering_metric_comparison.csv",

    index=False

)

# ==========================================================
# Comparison Plot
# ==========================================================

fig, ax1 = plt.subplots(figsize=(9,6))

ax1.plot(
    list(k_range),
    silhouette_scores,
    marker="o",
    linewidth=2,
    label="Silhouette Score"
)

ax1.set_xlabel("K")
ax1.set_ylabel("Silhouette Score")

ax2 = ax1.twinx()

ax2.plot(
    list(k_range),
    db_scores,
    marker="s",
    linewidth=2
)

ax2.set_ylabel("Davies-Bouldin Index")

plt.title("Silhouette vs Davies-Bouldin Comparison")

plt.tight_layout()

plt.savefig(
    "plots/silhouette_vs_db.png",
    dpi=300
)

plt.close()

print("Comparison Plot Saved")

print("\nTask 7 Completed Successfully")

# ==========================================================
# Task 8
# Baseline Model (K-Means)
# ==========================================================

print("\n" + "=" * 60)
print("Task 8 : Baseline Model (K-Means)")
print("=" * 60)

X = standard_scaled_df.copy()

# ==========================================================
# Train Final Model
# ==========================================================

kmeans_model = KMeans(
    n_clusters=best_k_silhouette,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans_model.fit_predict(X)

print("\nK-Means Model Trained Successfully")

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(
    kmeans_model,
    "models/kmeans_model.pkl"
)

print("K-Means Model Saved")

# ==========================================================
# Add Cluster Labels
# ==========================================================

cluster_df["Cluster"] = cluster_labels

cluster_df.to_csv(
    "reports/customer_clusters.csv",
    index=False
)

print("Cluster Labels Saved")

# ==========================================================
# Cluster Size
# ==========================================================

cluster_size = cluster_df["Cluster"].value_counts().sort_index()

print("\nCluster Size")
print(cluster_size)

cluster_size.to_csv(
    "reports/cluster_size.csv"
)

# ==========================================================
# Cluster Centers
# ==========================================================

centers = pd.DataFrame(

    kmeans_model.cluster_centers_,

    columns=cluster_df.columns[:-1]

)

centers.to_csv(

    "reports/cluster_centers.csv",

    index=False

)

print("\nCluster Centers Saved")

# ==========================================================
# Evaluation
# ==========================================================

sil_score = silhouette_score(
    X,
    cluster_labels
)

db_score = davies_bouldin_score(
    X,
    cluster_labels
)

print("\nSilhouette Score :", round(sil_score,4))
print("Davies-Bouldin Index :", round(db_score,4))

evaluation = pd.DataFrame({

    "Metric":[
        "Silhouette Score",
        "Davies-Bouldin Index"
    ],

    "Value":[
        sil_score,
        db_score
    ]

})

evaluation.to_csv(

    "reports/kmeans_evaluation.csv",

    index=False

)

# ==========================================================
# PCA Visualization
# ==========================================================

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(X)

pca_df = pd.DataFrame({

    "PC1": pca_result[:,0],

    "PC2": pca_result[:,1],

    "Cluster": cluster_labels

})

plt.figure(figsize=(8,6))

scatter = plt.scatter(

    pca_df["PC1"],

    pca_df["PC2"],

    c=pca_df["Cluster"],

    cmap="tab10"

)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("K-Means PCA Visualization")

plt.colorbar(scatter)

plt.tight_layout()

plt.savefig(

    "plots/kmeans_pca.png",

    dpi=300

)

plt.close()

# ==========================================================
# t-SNE Visualization
# ==========================================================

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

tsne_result = tsne.fit_transform(X)

tsne_df = pd.DataFrame({

    "TSNE1": tsne_result[:,0],

    "TSNE2": tsne_result[:,1],

    "Cluster": cluster_labels

})

plt.figure(figsize=(8,6))

scatter = plt.scatter(

    tsne_df["TSNE1"],

    tsne_df["TSNE2"],

    c=tsne_df["Cluster"],

    cmap="tab10"

)

plt.title("K-Means t-SNE Visualization")

plt.colorbar(scatter)

plt.tight_layout()

plt.savefig(

    "plots/kmeans_tsne.png",

    dpi=300

)

plt.close()

print("\nTask 8 Completed Successfully")

# ==========================================================
# Task 9
# Hierarchical Clustering
# ==========================================================

print("\n" + "="*60)
print("Task 9 : Hierarchical Clustering")
print("="*60)

from scipy.cluster.hierarchy import linkage, dendrogram

X = standard_scaled_df.copy()

# ==========================================================
# Dendrogram
# ==========================================================

print("\nGenerating Dendrogram...")

linked = linkage(
    X,
    method="ward"
)

plt.figure(figsize=(12,6))

dendrogram(
    linked,
    truncate_mode="level",
    p=5,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Hierarchical Clustering Dendrogram")

plt.xlabel("Samples")

plt.ylabel("Distance")

plt.tight_layout()

plt.savefig(
    "plots/dendrogram.png",
    dpi=300
)

plt.close()

print("Dendrogram Saved")

# ==========================================================
# Train Hierarchical Model
# ==========================================================

hierarchical_model = AgglomerativeClustering(
    n_clusters=best_k_silhouette
)

hierarchical_labels = hierarchical_model.fit_predict(X)

print("Hierarchical Model Trained")

# ==========================================================
# Save Labels
# ==========================================================

hierarchical_df = cluster_df.drop(columns=["Cluster"]).copy()

hierarchical_df["Cluster"] = hierarchical_labels

hierarchical_df.to_csv(
    "reports/hierarchical_clusters.csv",
    index=False
)

print("Cluster Labels Saved")

# ==========================================================
# Evaluation
# ==========================================================

hier_silhouette = silhouette_score(
    X,
    hierarchical_labels
)

hier_db = davies_bouldin_score(
    X,
    hierarchical_labels
)

print("\nSilhouette Score :", round(hier_silhouette,4))
print("Davies-Bouldin Index :", round(hier_db,4))

evaluation = pd.DataFrame({

    "Metric":[
        "Silhouette Score",
        "Davies-Bouldin Index"
    ],

    "Value":[
        hier_silhouette,
        hier_db
    ]

})

evaluation.to_csv(
    "reports/hierarchical_evaluation.csv",
    index=False
)

# ==========================================================
# Cluster Size
# ==========================================================

cluster_size = hierarchical_df["Cluster"].value_counts().sort_index()

cluster_size.to_csv(
    "reports/hierarchical_cluster_size.csv"
)

print("\nCluster Size")
print(cluster_size)

# ==========================================================
# PCA Visualization
# ==========================================================

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(X)

pca_df = pd.DataFrame({

    "PC1": pca_result[:,0],

    "PC2": pca_result[:,1],

    "Cluster": hierarchical_labels

})

plt.figure(figsize=(8,6))

scatter = plt.scatter(

    pca_df["PC1"],

    pca_df["PC2"],

    c=pca_df["Cluster"],

    cmap="tab10"

)

plt.title("Hierarchical PCA")

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.colorbar(scatter)

plt.tight_layout()

plt.savefig(
    "plots/hierarchical_pca.png",
    dpi=300
)

plt.close()

# ==========================================================
# Comparison with KMeans
# ==========================================================

comparison = pd.DataFrame({

    "Algorithm":[
        "KMeans",
        "Hierarchical"
    ],

    "Silhouette Score":[
        sil_score,
        hier_silhouette
    ],

    "Davies-Bouldin":[
        db_score,
        hier_db
    ]

})

comparison.to_csv(
    "reports/kmeans_vs_hierarchical.csv",
    index=False
)

print("\nComparison Report Saved")

print("\nTask 9 Completed Successfully")

# ==========================================================
# Task 10
# Gaussian Mixture Model (GMM)
# ==========================================================

print("\n" + "=" * 60)
print("Task 10 : Gaussian Mixture Model (GMM)")
print("=" * 60)

X = standard_scaled_df.copy()

# ==========================================================
# Train GMM
# ==========================================================

gmm_model = GaussianMixture(
    n_components=best_k_silhouette,
    covariance_type="full",
    random_state=42
)

gmm_labels = gmm_model.fit_predict(X)

print("\nGMM Model Trained Successfully")

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(
    gmm_model,
    "models/gmm_model.pkl"
)

print("GMM Model Saved")

# ==========================================================
# Save Cluster Labels
# ==========================================================

gmm_df = cluster_df.drop(columns=["Cluster"]).copy()

gmm_df["Cluster"] = gmm_labels

gmm_df.to_csv(
    "reports/gmm_clusters.csv",
    index=False
)

print("Cluster Labels Saved")

# ==========================================================
# Cluster Probability
# ==========================================================

cluster_probability = gmm_model.predict_proba(X)

probability_df = pd.DataFrame(
    cluster_probability,
    columns=[
        f"Cluster_{i}_Probability"
        for i in range(best_k_silhouette)
    ]
)

probability_df.to_csv(
    "reports/gmm_cluster_probability.csv",
    index=False
)

print("Cluster Probability Saved")

# ==========================================================
# Evaluation
# ==========================================================

gmm_silhouette = silhouette_score(
    X,
    gmm_labels
)

gmm_db = davies_bouldin_score(
    X,
    gmm_labels
)

gmm_aic = gmm_model.aic(X)

gmm_bic = gmm_model.bic(X)

print("\nSilhouette Score :", round(gmm_silhouette,4))
print("Davies-Bouldin :", round(gmm_db,4))
print("AIC :", round(gmm_aic,2))
print("BIC :", round(gmm_bic,2))

evaluation = pd.DataFrame({

    "Metric":[
        "Silhouette Score",
        "Davies-Bouldin Index",
        "AIC",
        "BIC"
    ],

    "Value":[
        gmm_silhouette,
        gmm_db,
        gmm_aic,
        gmm_bic
    ]

})

evaluation.to_csv(
    "reports/gmm_evaluation.csv",
    index=False
)

# ==========================================================
# Cluster Size
# ==========================================================

cluster_size = gmm_df["Cluster"].value_counts().sort_index()

cluster_size.to_csv(
    "reports/gmm_cluster_size.csv"
)

print("\nCluster Size")
print(cluster_size)

# ==========================================================
# PCA Visualization
# ==========================================================

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(X)

pca_df = pd.DataFrame({

    "PC1": pca_result[:,0],

    "PC2": pca_result[:,1],

    "Cluster": gmm_labels

})

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=pca_df["Cluster"],
    cmap="tab10"
)

plt.title("Gaussian Mixture Model (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)

plt.tight_layout()

plt.savefig(
    "plots/gmm_pca.png",
    dpi=300
)

plt.close()

# ==========================================================
# Probability Distribution
# ==========================================================

plt.figure(figsize=(10,6))

for i in range(best_k_silhouette):

    plt.hist(
        probability_df.iloc[:,i],
        bins=30,
        alpha=0.5,
        label=f"Cluster {i}"
    )

plt.legend()

plt.title("Cluster Probability Distribution")

plt.tight_layout()

plt.savefig(
    "plots/gmm_probability_distribution.png",
    dpi=300
)

plt.close()

print("Probability Distribution Saved")

print("\nTask 10 Completed Successfully")

# ==========================================================
# Task 11
# DBSCAN Clustering
# ==========================================================

print("\n" + "="*60)
print("Task 11 : DBSCAN Clustering")
print("="*60)

X = standard_scaled_df.copy()

# ----------------------------------------------------------
# Train DBSCAN
# ----------------------------------------------------------

eps_value = 0.8
min_samples_value = 8

dbscan_model = DBSCAN(
    eps=eps_value,
    min_samples=min_samples_value
)

dbscan_labels = dbscan_model.fit_predict(X)

print("\nDBSCAN Model Trained Successfully")

# ----------------------------------------------------------
# Save Labels
# ----------------------------------------------------------

dbscan_df = cluster_df.drop(columns=["Cluster"]).copy()
dbscan_df["Cluster"] = dbscan_labels

dbscan_df.to_csv(
    "reports/dbscan_clusters.csv",
    index=False
)

# ----------------------------------------------------------
# Number of Clusters
# ----------------------------------------------------------

num_clusters = len(set(dbscan_labels))

if -1 in dbscan_labels:
    num_clusters -= 1

noise_points = list(dbscan_labels).count(-1)

print(f"\nClusters Found : {num_clusters}")
print(f"Noise Points   : {noise_points}")

# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------

if num_clusters > 1:

    mask = dbscan_labels != -1

    sil = silhouette_score(
        X[mask],
        dbscan_labels[mask]
    )

    db = davies_bouldin_score(
        X[mask],
        dbscan_labels[mask]
    )

else:

    sil = np.nan
    db = np.nan

evaluation = pd.DataFrame({

    "Metric":[
        "Clusters",
        "Noise Points",
        "Silhouette Score",
        "Davies-Bouldin Index"
    ],

    "Value":[
        num_clusters,
        noise_points,
        sil,
        db
    ]

})

evaluation.to_csv(
    "reports/dbscan_evaluation.csv",
    index=False
)

print(evaluation)

# ----------------------------------------------------------
# PCA Visualization
# ----------------------------------------------------------

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(X)

plt.figure(figsize=(8,6))

scatter = plt.scatter(

    pca_result[:,0],
    pca_result[:,1],

    c=dbscan_labels,

    cmap="tab20"

)

plt.title("DBSCAN Clustering")

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)

plt.tight_layout()

plt.savefig(
    "plots/dbscan_pca.png",
    dpi=300
)

plt.close()

print("\nTask 11 Completed Successfully")

# ==========================================================
# Task 12
# Algorithm Comparison
# ==========================================================

print("\n" + "="*60)
print("Task 12 : Algorithm Comparison")
print("="*60)

comparison = pd.DataFrame({

    "Algorithm":[
        "KMeans",
        "Hierarchical",
        "Gaussian Mixture",
        "DBSCAN"
    ],

    "Silhouette Score":[
        sil_score,
        hier_silhouette,
        gmm_silhouette,
        sil
    ],

    "Davies-Bouldin":[
        db_score,
        hier_db,
        gmm_db,
        db
    ],

    "Clusters":[
        best_k_silhouette,
        best_k_silhouette,
        best_k_silhouette,
        num_clusters
    ]

})

comparison.to_csv(
    "reports/algorithm_comparison.csv",
    index=False
)

print(comparison)

# ==========================================================
# Best Algorithm
# ==========================================================

best_algorithm = comparison.sort_values(
    by="Silhouette Score",
    ascending=False
).iloc[0]["Algorithm"]

print("\nBest Algorithm :", best_algorithm)

# ==========================================================
# Task 13
# Cluster Profiling
# ==========================================================

print("\n" + "="*60)
print("Task 13 : Cluster Profiling")
print("="*60)

profile = cluster_df.groupby("Cluster").mean()

profile.to_csv(
    "reports/cluster_profile.csv"
)

print(profile)

# ----------------------------------------------------------
# Business Labels
# ----------------------------------------------------------

cluster_names = {}

for cluster in profile.index:

    income_value = profile.loc[cluster]["Income"]

    if income_value > profile["Income"].mean():

        cluster_names[cluster] = "High Value Customers"

    else:

        cluster_names[cluster] = "Regular Customers"

cluster_summary = pd.DataFrame({

    "Cluster":list(cluster_names.keys()),

    "Business_Name":list(cluster_names.values())

})

cluster_summary.to_csv(
    "reports/business_cluster_names.csv",
    index=False
)

print(cluster_summary)

# ==========================================================
# Task 14
# Cluster Visualization
# ==========================================================

print("\n" + "="*60)
print("Task 14 : Cluster Visualization")
print("="*60)

X = standard_scaled_df.copy()

# ----------------------------------------------------------
# PCA
# ----------------------------------------------------------

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(X)

plt.figure(figsize=(8,6))

scatter = plt.scatter(

    pca_result[:,0],

    pca_result[:,1],

    c=cluster_labels,

    cmap="tab10"

)

plt.colorbar(scatter)

plt.title("Final Cluster Visualization (PCA)")

plt.tight_layout()

plt.savefig(
    "plots/final_pca_clusters.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# t-SNE
# ----------------------------------------------------------

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

tsne_result = tsne.fit_transform(X)

plt.figure(figsize=(8,6))

scatter = plt.scatter(

    tsne_result[:,0],

    tsne_result[:,1],

    c=cluster_labels,

    cmap="tab10"

)

plt.colorbar(scatter)

plt.title("Final Cluster Visualization (t-SNE)")

plt.tight_layout()

plt.savefig(
    "plots/final_tsne_clusters.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# Heatmap
# ----------------------------------------------------------

plt.figure(figsize=(10,6))

sns.heatmap(
    profile,
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)

plt.title("Cluster Heatmap")

plt.tight_layout()

plt.savefig(
    "plots/cluster_heatmap.png",
    dpi=300
)

plt.close()

print("\nTask 12, 13 and 14 Completed Successfully")

# ==========================================================
# Task 15
# Cluster Stability Analysis
# ==========================================================

print("\n" + "="*60)
print("Task 15 : Cluster Stability Analysis")
print("="*60)

from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample

X = standard_scaled_df.copy()

stability_scores = []

random_states = [42, 100, 200, 300, 500]

base_model = KMeans(
    n_clusters=best_k_silhouette,
    random_state=42,
    n_init=10
)

base_labels = base_model.fit_predict(X)

for state in random_states:

    model = KMeans(
        n_clusters=best_k_silhouette,
        random_state=state,
        n_init=10
    )

    labels = model.fit_predict(X)

    ari = adjusted_rand_score(
        base_labels,
        labels
    )

    stability_scores.append(ari)

stability_df = pd.DataFrame({

    "Random_State": random_states,

    "Adjusted_Rand_Index": stability_scores

})

stability_df.to_csv(
    "reports/stability_analysis.csv",
    index=False
)

print(stability_df)

# ==========================================================
# Bootstrap Stability
# ==========================================================

bootstrap_scores = []

for i in range(5):

    sample = resample(
        X,
        replace=True,
        random_state=i
    )

    model = KMeans(
        n_clusters=best_k_silhouette,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(sample)

    score = silhouette_score(
        sample,
        labels
    )

    bootstrap_scores.append(score)

bootstrap_df = pd.DataFrame({

    "Iteration": range(1,6),

    "Silhouette": bootstrap_scores

})

bootstrap_df.to_csv(
    "reports/bootstrap_stability.csv",
    index=False
)

print(bootstrap_df)

print("\nTask 15 Completed Successfully")

# ==========================================================
# Task 16
# Final Model Selection
# ==========================================================

print("\n" + "="*60)
print("Task 16 : Final Model Selection")
print("="*60)

final_algorithm = comparison.sort_values(
    by="Silhouette Score",
    ascending=False
).iloc[0]

print(final_algorithm)

final_model_report = pd.DataFrame({

    "Best Algorithm":[final_algorithm["Algorithm"]],

    "Silhouette":[final_algorithm["Silhouette Score"]],

    "Davies-Bouldin":[final_algorithm["Davies-Bouldin"]]

})

final_model_report.to_csv(
    "reports/final_model_selection.csv",
    index=False
)

print("\nTask 16 Completed Successfully")

# ==========================================================
# Task 17
# Save Production Pipeline
# ==========================================================

print("\n" + "="*60)
print("Task 17 : Save Production Pipeline")
print("="*60)

production_pipeline = {

    "Selected Features": available_features,

    "Scaler": standard_scaler,

    "Best K": best_k_silhouette,

    "KMeans Model": kmeans_model,

    "Hierarchical": hierarchical_model,

    "Gaussian Mixture": gmm_model,

    "DBSCAN": dbscan_model

}

joblib.dump(

    production_pipeline,

    "pipeline/customer_segmentation_pipeline.pkl"

)

print("Production Pipeline Saved")

print("\nTask 17 Completed Successfully")

# ==========================================================
# Task 18
# Final Documentation
# ==========================================================

print("\n" + "="*60)
print("Task 18 : Final Documentation")
print("="*60)

report = f"""
==================================================
Customer Segmentation Project Report
==================================================

Problem Statement
-----------------
Customer segmentation using unsupervised learning.

Dataset
-------
Marketing Campaign Dataset

Algorithms
----------
1. KMeans
2. Hierarchical Clustering
3. Gaussian Mixture Model
4. DBSCAN

Selected Features
-----------------
{available_features}

Best Algorithm
--------------
{final_algorithm['Algorithm']}

Silhouette Score
----------------
{final_algorithm['Silhouette Score']:.4f}

Davies-Bouldin Index
--------------------
{final_algorithm['Davies-Bouldin']:.4f}

Generated Files
---------------
Reports
Plots
Models
Pipeline

Project Status
--------------
Completed Successfully
"""

with open(
    "reports/final_project_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

print(report)

print("\n" + "="*60)
print("ALL 18 TASKS COMPLETED SUCCESSFULLY")
print("="*60)
