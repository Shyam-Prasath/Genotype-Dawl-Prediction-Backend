import os
import pandas as pd

# ===========================
# Setup paths
# ===========================

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# File paths
file_main = os.path.join(project_root, "cross_pop_results.csv")
file_rf = os.path.join(project_root, "cross_pop_results_rf_fixed.csv")

# ===========================
# Load data
# ===========================

print("Loading files...")

df_main = pd.read_csv(file_main)
df_rf = pd.read_csv(file_rf)

print("Main file shape:", df_main.shape)
print("RF file shape:", df_rf.shape)

# ===========================
# Merge
# ===========================

df_all = pd.concat([df_main, df_rf], ignore_index=True)

# ===========================
# Save merged file
# ===========================

output_path = os.path.join(project_root, "cross_pop_results_all_models.csv")
df_all.to_csv(output_path, index=False)

print("\nMerged successfully!")
print("Saved as:", output_path)
print("Total rows:", df_all.shape[0])

# ===========================
# Quick check
# ===========================

print("\nModels present:", df_all["model"].unique())
print("\nPreview:")
print(df_all.head())