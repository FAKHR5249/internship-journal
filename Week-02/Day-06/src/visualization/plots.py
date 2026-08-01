import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# Paths
# ==========================================

REPORT_PATH = Path("reports/predictions.csv")
OUTPUT_DIR = Path("reports")

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# Load Predictions
# ==========================================

df = pd.read_csv(REPORT_PATH)

print("Predictions Loaded Successfully")

# ==========================================
# Scatter Plot
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["Actual"],
    df["Predicted"],
    alpha=0.5
)

plt.xlabel("Actual Trip Duration")
plt.ylabel("Predicted Trip Duration")
plt.title("Actual vs Predicted")

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# Histogram
# ==========================================

plt.figure(figsize=(8,6))

plt.hist(
    df["Actual"],
    bins=30,
    alpha=0.7,
    label="Actual"
)

plt.hist(
    df["Predicted"],
    bins=30,
    alpha=0.7,
    label="Predicted"
)

plt.legend()

plt.title("Distribution of Actual and Predicted")

plt.savefig(
    OUTPUT_DIR / "distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Graphs Saved Successfully")