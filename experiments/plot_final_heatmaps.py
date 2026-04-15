import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Setup paths
# ===========================

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

file_path = os.path.join(project_root, "cross_pop_results_all_models.csv")

# ===========================
# Load data
# ===========================

df = pd.read_csv(file_path)

# Compute mean correlation
mean_df = (
    df.groupby(["model", "source_panel", "target_panel"])["correlation"]
    .mean()
    .reset_index()
)

panels = sorted(df["source_panel"].unique())
models = ["Ridge", "MLP", "RandomForest"]  # fixed order

print("Panels:", panels)
print("Models:", models)

# ===========================
# Heatmap settings (IMPORTANT)
# ===========================

VMIN = -0.4   # consistent color scale
VMAX = 0.9

# ===========================
# Plot heatmaps
# ===========================

for model in models:

    subset = mean_df[mean_df["model"] == model]

    matrix = np.zeros((len(panels), len(panels)))

    for i, src in enumerate(panels):
        for j, tgt in enumerate(panels):

            if src == tgt:
                matrix[i, j] = np.nan
            else:
                val = subset[
                    (subset["source_panel"] == src) &
                    (subset["target_panel"] == tgt)
                ]["correlation"].values

                matrix[i, j] = val[0] if len(val) > 0 else np.nan

    # ===========================
    # Plot
    # ===========================

    plt.figure(figsize=(6, 5))

    im = plt.imshow(matrix, vmin=VMIN, vmax=VMAX)

    plt.xticks(range(len(panels)), panels)
    plt.yticks(range(len(panels)), panels)

    plt.xlabel("Target Panel", fontsize=11)
    plt.ylabel("Source Panel", fontsize=11)

    # Proper IEEE title
    if model == "RandomForest":
        title_name = "Random Forest"
    else:
        title_name = model

    plt.title(f"Cross-Population Prediction Performance ({title_name})", fontsize=12)

    # Annotate values
    for i in range(len(panels)):
        for j in range(len(panels)):
            if not np.isnan(matrix[i, j]):
                plt.text(
                    j, i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black"
                )

    cbar = plt.colorbar(im)
    cbar.set_label("Pearson Correlation (r)", fontsize=10)

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(
        project_root,
        f"Figure_Heatmap_{model}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

print("\n✅ All heatmaps generated successfully!")