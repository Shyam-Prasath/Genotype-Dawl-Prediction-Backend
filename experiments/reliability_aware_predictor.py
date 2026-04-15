import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Setup
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===========================
# Load Dataset
# ===========================
data_path = os.path.join(project_root, "final_merged_dataset.txt")
df = pd.read_csv(data_path, sep="\t")

X_all = df.drop(columns=["Base_ID", "Full_name", "Panel", "PH", "EH"]).values
y_all = df["PH"].values
panel_labels = df["Panel"].values

# ===========================
# MLP with Dropout
# ===========================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

# ===========================
# Monte Carlo Prediction
# ===========================
def mc_dropout_predict(model, X, n_samples=30):
    model.train()  # Keep dropout active
    preds = []

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    for _ in range(n_samples):
        with torch.no_grad():
            prediction = model(X_tensor).cpu().numpy().flatten()
            preds.append(prediction)

    preds = np.array(preds)

    mean_prediction = preds.mean(axis=0)
    uncertainty = preds.std(axis=0)

    return mean_prediction, uncertainty

# ===========================
# Train on ASSO + NCRIPS
# Test on USP
# ===========================
target_panel = "USP"

train_mask = panel_labels != target_panel
test_mask = panel_labels == target_panel

X_train = X_all[train_mask]
y_train = y_all[train_mask]
X_test = X_all[test_mask]
y_test = y_all[test_mask]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLP(X_train.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)

# Training
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Training completed.")

# ===========================
# Monte Carlo Inference
# ===========================
mean_pred, uncertainty = mc_dropout_predict(model, X_test, n_samples=30)

corr, _ = pearsonr(y_test, mean_pred)

print("\nUSP Test Performance")
print("Correlation:", round(corr,4))
print("Mean Uncertainty:", round(np.mean(uncertainty),4))
print("Max Uncertainty:", round(np.max(uncertainty),4))

# ===========================
# Reliability Validation
# ===========================

absolute_error = np.abs(y_test - mean_pred)

error_uncertainty_corr, _ = pearsonr(uncertainty, absolute_error)

print("\nUncertainty vs Absolute Error Correlation:",
    round(error_uncertainty_corr,4))