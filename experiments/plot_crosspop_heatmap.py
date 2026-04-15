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
sys.path.append(project_root)

# ===========================
# Load Results
# ===========================
results_path = os.path.join(project_root, "cross_pop_results.csv")
df = pd.read_csv(results_path)

print("CSV Columns:", df.columns)

# Compute mean correlation across seeds
mean_df = (
    df.groupby(["model", "source_panel", "target_panel"])["correlation"]
      .mean()
      .reset_index()
)

panels = sorted(df["source_panel"].unique())
models = sorted(df["model"].unique())

# ===========================
# Build Heatmap For Each Model
# ===========================
for model in models:

    matrix = np.full((len(panels), len(panels)), np.nan)

    for i, target in enumerate(panels):
        for j, source in enumerate(panels):

            if source == target:
                continue

            value = mean_df[
                (mean_df["model"] == model) &
                (mean_df["source_panel"] == source) &
                (mean_df["target_panel"] == target)
            ]["correlation"].values

            if len(value) > 0:
                matrix[i, j] = value[0]

    # ===========================
    # Plot
    # ===========================
    plt.figure(figsize=(6,5))
    im = plt.imshow(matrix)

    plt.xticks(range(len(panels)), panels)
    plt.yticks(range(len(panels)), panels)

    plt.xlabel("Source Panel")
    plt.ylabel("Target Panel")
    plt.title(f"Cross-Population PH Correlation ({model})")

    # Annotate values
    for i in range(len(panels)):
        for j in range(len(panels)):
            if not np.isnan(matrix[i, j]):
                plt.text(j, i, f"{matrix[i,j]:.2f}",
                         ha="center", va="center")

    plt.colorbar(im)
    plt.tight_layout()

    save_path = os.path.join(project_root,
                             f"Figure4_Heatmap_{model}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)