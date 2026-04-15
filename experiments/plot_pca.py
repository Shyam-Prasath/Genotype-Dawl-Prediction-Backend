import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Setup
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# Load dataset
data_path = os.path.join(project_root, "final_merged_dataset.txt")
df = pd.read_csv(data_path, sep="\t")

X = df.drop(columns=["Base_ID", "Full_name", "Panel", "PH", "EH"]).values
panels = df["Panel"].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_

# Plot
plt.figure(figsize=(8,6))

unique_panels = np.unique(panels)

for panel in unique_panels:
    idx = panels == panel
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        alpha=0.6,
        s=10,
        label=panel
    )

plt.xlabel(f"Principal Component 1 ({explained_var[0]*100:.1f}% variance)")
plt.ylabel(f"Principal Component 2 ({explained_var[1]*100:.1f}% variance)")
plt.title("PCA Projection of Genomic Data Across Panels")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(project_root, "Figure1_PCA.png"), dpi=300)
plt.show()

print("Figure1_PCA.png saved.")