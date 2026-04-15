import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

# Setup
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# Load dataset
data_path = os.path.join(project_root, "final_merged_dataset.txt")
df = pd.read_csv(data_path, sep="\t")

X = df.drop(columns=["Base_ID", "Full_name", "Panel", "PH", "EH"]).values
panels = df["Panel"].values

# Standardize + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)

# Compute centroids
unique_panels = np.unique(panels)
centroids = {}

for panel in unique_panels:
    centroids[panel] = np.mean(X_pca[panels == panel], axis=0)

# Compute distances
pairs = []
values = []

for i in range(len(unique_panels)):
    for j in range(i+1, len(unique_panels)):
        p1 = unique_panels[i]
        p2 = unique_panels[j]
        dist = euclidean(centroids[p1], centroids[p2])
        pairs.append(f"{p1} vs {p2}")
        values.append(dist)

# Plot
plt.figure(figsize=(8,6))
bars = plt.bar(pairs, values)

for i, v in enumerate(values):
    plt.text(i, v *1.02 , f"{v:.1f}", ha='center')

plt.ylabel("Genetic Distance (PCA Space)")
plt.title("Genetic Centroid Distances Between Panels")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(os.path.join(project_root, "Figure2_GeneticDistance.png"), dpi=300)
plt.show()

print("Figure2_GeneticDistance.png saved.")