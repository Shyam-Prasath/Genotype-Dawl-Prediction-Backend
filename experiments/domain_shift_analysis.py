import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

# Setup path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# ===========================
# Load Dataset
# ===========================
data_path = os.path.join(project_root, "final_merged_dataset.txt")
df = pd.read_csv(data_path, sep="\t")

X = df.drop(columns=["Base_ID", "Full_name", "Panel", "PH", "EH"]).values
panels = df["Panel"].values

# ===========================
# Standardize SNP Matrix
# ===========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# PCA Projection
# ===========================
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance (First 5 PCs):")
print(pca.explained_variance_ratio_[:5])
print("Total Variance Explained (20 PCs):",
      np.sum(pca.explained_variance_ratio_))

# ===========================
# Compute Panel Centroids
# ===========================
unique_panels = np.unique(panels)
centroids = {}

for panel in unique_panels:
    panel_points = X_pca[panels == panel]
    centroids[panel] = np.mean(panel_points, axis=0)

# ===========================
# Compute Pairwise Distances
# ===========================
print("\nGenetic Distance Between Panels (PCA Space):\n")

for i in range(len(unique_panels)):
    for j in range(i+1, len(unique_panels)):
        p1 = unique_panels[i]
        p2 = unique_panels[j]

        dist = euclidean(centroids[p1], centroids[p2])

        print(f"{p1} vs {p2} Distance: {round(dist, 4)}")