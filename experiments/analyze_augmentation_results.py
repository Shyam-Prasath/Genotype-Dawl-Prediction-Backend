import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# Setup
# ===========================
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

data_path = os.path.join(project_root, "augmentation_results_usp.csv")
df = pd.read_csv(data_path)

print("Total rows:", len(df))

# ===========================
# 1️⃣ Compute Mean ± Std
# ===========================
summary = (
    df.groupby("augmentation_pct")
    .agg(
        ridge_mean=("ridge_corr", "mean"),
        ridge_std=("ridge_corr", "std"),
        mlp_mean=("mlp_corr", "mean"),
        mlp_std=("mlp_corr", "std"),
    )
    .reset_index()
)

print("\nAugmentation Summary:")
print(summary)

summary_path = os.path.join(project_root, "augmentation_summary_usp.csv")
summary.to_csv(summary_path, index=False)
print("\nSaved augmentation_summary_usp.csv")

# ===========================
# 2️⃣ Plot Final IEEE Figure
# ===========================
plt.figure(figsize=(8,6))

plt.errorbar(
    summary["augmentation_pct"],
    summary["ridge_mean"],
    yerr=summary["ridge_std"],
    marker="o",
    capsize=4,
    label="Ridge"
)

plt.errorbar(
    summary["augmentation_pct"],
    summary["mlp_mean"],
    yerr=summary["mlp_std"],
    marker="o",
    capsize=4,
    label="MLP"
)

plt.xlabel("Target Population Added (%)")
plt.ylabel("Cross-Population PH Correlation")
plt.title("Performance Recovery via Target Augmentation (USP)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

figure_path = os.path.join(project_root, "Figure_Augmentation_USP.png")
plt.savefig(figure_path, dpi=300)
plt.show()

print("\nSaved Figure_Augmentation_USP.png")