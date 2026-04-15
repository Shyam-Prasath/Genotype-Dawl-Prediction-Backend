import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Setup paths
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===========================
# Load Data
# ===========================
data_path = os.path.join(project_root, "final_merged_dataset.txt")
df = pd.read_csv(data_path, sep="\t")

X = df.drop(columns=["Base_ID", "Full_name", "Panel", "PH", "EH"]).values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# ===========================
# Masking Function
# ===========================
def mask_input(x, mask_ratio=0.15):
    mask = torch.rand_like(x) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked, mask

# ===========================
# Autoencoder Model
# ===========================
class SNPEncoder(nn.Module):
    def __init__(self, input_dim):
        super(SNPEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# ===========================
# Training
# ===========================
model = SNPEncoder(X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 20
batch_size = 128

print("\nStarting Self-Supervised Pretraining...\n")

for epoch in range(epochs):

    perm = torch.randperm(X_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_tensor.size(0), batch_size):
        indices = perm[i:i+batch_size]
        batch = X_tensor[indices]

        masked_batch, mask = mask_input(batch)

        optimizer.zero_grad()
        outputs = model(masked_batch)

        loss = criterion(outputs[mask], batch[mask])
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save encoder weights
torch.save(model.encoder.state_dict(),
           os.path.join(project_root, "pretrained_encoder.pth"))

print("\nPretraining Completed. Encoder Saved.")