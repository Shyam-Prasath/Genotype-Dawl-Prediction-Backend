import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# ===========================
# Setup
# ===========================
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===========================
# MLP Model
# ===========================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_mlp(X_train, y_train, X_test, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    epochs = 50
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()

    return preds

# ===========================
# Load Dataset
# ===========================
data_path = os.path.join(project_root, "final_merged_dataset.txt")
df = pd.read_csv(data_path, sep="\t")

panels = df["Panel"].unique()
print("Available Panels:", panels)

X_all = df.drop(columns=["Base_ID", "Full_name", "Panel", "PH", "EH"]).values
y_all = df["PH"].values
panel_labels = df["Panel"].values

# ===========================
# Global PCA for distance
# ===========================
scaler_global = StandardScaler()
X_scaled_global = scaler_global.fit_transform(X_all)

pca = PCA(n_components=20)
X_pca_global = pca.fit_transform(X_scaled_global)

# ===========================
# Cross-Population Evaluation
# ===========================
seeds = [42, 123, 999, 2024, 7]

panel_results = []
degradation_results = []

for source_panel in panels:
    for target_panel in panels:

        if source_panel == target_panel:
            continue

        print("\n==============================")
        print(f"Train: {source_panel} → Test: {target_panel}")
        print("==============================")

        train_mask = panel_labels == source_panel
        test_mask = panel_labels == target_panel

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]

        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        # PCA subsets
        X_train_pca = X_pca_global[train_mask]
        X_test_pca = X_pca_global[test_mask]

        train_centroid = np.mean(X_train_pca, axis=0)

        # Standardize based on source only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for seed in seeds:

            # Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled, y_train)
            preds_ridge = ridge.predict(X_test_scaled)

            corr_ridge, _ = pearsonr(y_test, preds_ridge)

            panel_results.append({
                "model": "Ridge",
                "source_panel": source_panel,
                "target_panel": target_panel,
                "seed": seed,
                "correlation": corr_ridge
            })

            # Genotype-level degradation
            abs_errors = np.abs(y_test - preds_ridge)

            for i in range(len(X_test_pca)):
                distance = np.linalg.norm(X_test_pca[i] - train_centroid)

                degradation_results.append({
                    "model": "Ridge",
                    "source_panel": source_panel,
                    "target_panel": target_panel,
                    "distance": distance,
                    "abs_error": abs_errors[i]
                })

            # MLP
            preds_mlp = train_mlp(X_train_scaled, y_train, X_test_scaled, seed)
            corr_mlp, _ = pearsonr(y_test, preds_mlp)

            panel_results.append({
                "model": "MLP",
                "source_panel": source_panel,
                "target_panel": target_panel,
                "seed": seed,
                "correlation": corr_mlp
            })

            abs_errors_mlp = np.abs(y_test - preds_mlp)

            for i in range(len(X_test_pca)):
                distance = np.linalg.norm(X_test_pca[i] - train_centroid)

                degradation_results.append({
                    "model": "MLP",
                    "source_panel": source_panel,
                    "target_panel": target_panel,
                    "distance": distance,
                    "abs_error": abs_errors_mlp[i]
                })

            print(f"Seed {seed} | Ridge: {corr_ridge:.4f} | MLP: {corr_mlp:.4f}")

# ===========================
# Save Results
# ===========================
panel_df = pd.DataFrame(panel_results)
panel_df.to_csv(os.path.join(project_root, "cross_pop_results.csv"), index=False)

degradation_df = pd.DataFrame(degradation_results)
degradation_df.to_csv(os.path.join(project_root, "genotype_degradation_results.csv"), index=False)

print("\nSaved:")
print(" - cross_pop_results.csv")
print(" - genotype_degradation_results.csv")
print("Total genotype rows:", len(degradation_df))