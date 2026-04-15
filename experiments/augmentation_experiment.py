import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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

X_all = df.drop(columns=["Base_ID", "Full_name", "Panel", "PH", "EH"]).values
y_all = df["PH"].values
panels = df["Panel"].values

# Focus on USP
target_panel = "USP"

train_mask_full = panels != target_panel
target_mask = panels == target_panel

X_source = X_all[train_mask_full]
y_source = y_all[train_mask_full]

X_target_full = X_all[target_mask]
y_target_full = y_all[target_mask]

print("Source size:", len(X_source))
print("Target size:", len(X_target_full))

# ===========================
# Augmentation Percentages
# ===========================
percentages = [0, 5, 10, 20, 30]
seeds = [42, 123, 999, 2024, 7]

results = []

for pct in percentages:

    print(f"\nAugmentation: {pct}%")

    for seed in seeds:

        np.random.seed(seed)

        if pct == 0:
            X_train = X_source
            y_train = y_source
        else:
            n_add = int(len(X_target_full) * pct / 100)

            idx = np.random.choice(len(X_target_full), n_add, replace=False)

            X_aug = X_target_full[idx]
            y_aug = y_target_full[idx]

            X_train = np.vstack([X_source, X_aug])
            y_train = np.concatenate([y_source, y_aug])

        # Remaining USP samples used for testing
        if pct == 0:
            X_test = X_target_full
            y_test = y_target_full
        else:
            mask = np.ones(len(X_target_full), dtype=bool)
            mask[idx] = False
            X_test = X_target_full[mask]
            y_test = y_target_full[mask]

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ------------------
        # Ridge
        # ------------------
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        preds_ridge = ridge.predict(X_test)
        corr_ridge, _ = pearsonr(y_test, preds_ridge)

        # ------------------
        # MLP
        # ------------------
        preds_mlp = train_mlp(X_train, y_train, X_test, seed)
        corr_mlp, _ = pearsonr(y_test, preds_mlp)

        results.append({
            "augmentation_pct": pct,
            "seed": seed,
            "ridge_corr": corr_ridge,
            "mlp_corr": corr_mlp
        })

        print(f"Seed {seed} | Ridge: {corr_ridge:.4f} | MLP: {corr_mlp:.4f}")

# ===========================
# Save Results
# ===========================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(project_root, "augmentation_results_usp.csv"), index=False)

print("\nSaved augmentation_results_usp.csv")