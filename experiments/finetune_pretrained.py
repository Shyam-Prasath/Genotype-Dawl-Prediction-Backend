import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Setup paths
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
y_ph_all = df["PH"].values
y_eh_all = df["EH"].values
panel_labels = df["Panel"].values

panels = np.unique(panel_labels)
print("Panels:", panels)

# ===========================
# Model Definition
# ===========================
class PretrainedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Same encoder structure as pretraining
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.head = nn.Linear(256, 1)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)

# ===========================
# Load Pretrained Weights
# ===========================
def load_pretrained_encoder(model):
    encoder_path = os.path.join(project_root, "pretrained_encoder.pth")
    pretrained_weights = torch.load(encoder_path, map_location=device)
    model.encoder.load_state_dict(pretrained_weights)

# ===========================
# Training Function
# ===========================
def train_and_evaluate(X_train, y_train, X_test, y_test):

    model = PretrainedModel(X_train.shape[1]).to(device)
    load_pretrained_encoder(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()

    corr, _ = pearsonr(y_test, preds)
    return corr

# ===========================
# Cross-Population Evaluation
# ===========================
for test_panel in panels:

    print("\n==============================")
    print("Test Population:", test_panel)
    print("==============================")

    train_mask = panel_labels != test_panel
    test_mask = panel_labels == test_panel

    X_train, X_test = X_all[train_mask], X_all[test_mask]
    y_train_ph, y_test_ph = y_ph_all[train_mask], y_ph_all[test_mask]
    y_train_eh, y_test_eh = y_eh_all[train_mask], y_eh_all[test_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # PH
    corr_ph = train_and_evaluate(X_train, y_train_ph, X_test, y_test_ph)

    # EH
    corr_eh = train_and_evaluate(X_train, y_train_eh, X_test, y_test_eh)

    print("Pretrained MLP - PH:", round(corr_ph,4))
    print("Pretrained MLP - EH:", round(corr_eh,4))