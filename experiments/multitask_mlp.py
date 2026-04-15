import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tqdm import tqdm

# Allow utils import
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from experiments.utils import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===========================
# Multi-Task MLP Model
# ===========================
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskMLP, self).__init__()

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Two output heads
        self.head_ph = nn.Linear(256, 1)
        self.head_eh = nn.Linear(256, 1)

    def forward(self, x):
        features = self.shared(x)
        ph_out = self.head_ph(features)
        eh_out = self.head_eh(features)
        return ph_out, eh_out


# ===========================
# Training Function
# ===========================
def train_fold(X_train, y_train_ph, y_train_eh,
               X_val, y_val_ph, y_val_eh,
               epochs=50):

    model = MultiTaskMLP(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_ph = torch.tensor(y_train_ph, dtype=torch.float32).view(-1, 1).to(device)
    y_train_eh = torch.tensor(y_train_eh, dtype=torch.float32).view(-1, 1).to(device)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_ph = torch.tensor(y_val_ph, dtype=torch.float32).view(-1, 1).to(device)
    y_val_eh = torch.tensor(y_val_eh, dtype=torch.float32).view(-1, 1).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out_ph, out_eh = model(X_train)

        loss_ph = criterion(out_ph, y_train_ph)
        loss_eh = criterion(out_eh, y_train_eh)

        loss = loss_ph + loss_eh
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_ph, pred_eh = model(X_val)

        pred_ph = pred_ph.cpu().numpy().flatten()
        pred_eh = pred_eh.cpu().numpy().flatten()

        corr_ph, _ = pearsonr(y_val_ph.cpu().numpy().flatten(), pred_ph)
        corr_eh, _ = pearsonr(y_val_eh.cpu().numpy().flatten(), pred_eh)

    return corr_ph, corr_eh


# ===========================
# Cross Validation
# ===========================
X, y_ph, y_eh = load_data()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

ph_corrs = []
eh_corrs = []

print("\nStarting Multi-Task Evaluation...\n")

for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=5)):

    print(f"\n--- Fold {fold+1} ---")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_ph, y_val_ph = y_ph[train_idx], y_ph[val_idx]
    y_train_eh, y_val_eh = y_eh[train_idx], y_eh[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    corr_ph, corr_eh = train_fold(
        X_train, y_train_ph, y_train_eh,
        X_val, y_val_ph, y_val_eh
    )

    print("PH Correlation:", corr_ph)
    print("EH Correlation:", corr_eh)

    ph_corrs.append(corr_ph)
    eh_corrs.append(corr_eh)

print("\nMulti-Task MLP Results")
print("Mean PH Correlation:", np.mean(ph_corrs))
print("Mean EH Correlation:", np.mean(eh_corrs))