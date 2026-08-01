import pandas as pd
import os

# ==========================
# Load Dataset
# ==========================

file_path = "01_Raw_Data/marketing_campaign.csv"

# Load Dataset
df = pd.read_csv(file_path, sep="\t")

# ==========================
# Basic Information
# ==========================

rows = df.shape[0]
columns = df.shape[1]

file_size = os.path.getsize(file_path) / 1024  # KB

file_type = ".csv"

encoding = "UTF-8 (Assumed)"

# ==========================
# Print Report
# ==========================

print("=" * 45)
print("DATASET INSPECTION REPORT")
print("=" * 45)

print(f"Rows          : {rows}")
print(f"Columns       : {columns}")
print(f"Dataset Size  : {file_size:.2f} KB")
print(f"File Type     : {file_type}")
print(f"Encoding      : {encoding}")