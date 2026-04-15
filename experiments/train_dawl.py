import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# =========================
# 1. LOAD DATA (FINAL MERGED)
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "final_merged_dataset.txt")

print("📂 DATA PATH:", DATA_PATH)

# 🔥 Robust loading
data = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    engine="python",
    on_bad_lines="skip"
)

print("\n🔍 DATA PREVIEW:")
print(data.head())

print("\n✅ DATA SHAPE:", data.shape)

# =========================
# 2. CLEAN DATA
# =========================

# Ensure Base_ID exists
if "Base_ID" not in data.columns:
    data.rename(columns={data.columns[0]: "Base_ID"}, inplace=True)

# Convert numeric safely
for col in data.columns:
    if col not in ["Base_ID", "Panel"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Fill missing values
data = data.fillna(0)

# =========================
# 3. PREPARE FEATURES (X) & TARGET (y)
# =========================

# Select numeric columns only (SNP features)
X = data.select_dtypes(include=[np.number])

# Remove target columns
X = X.drop(columns=["PH", "EH"], errors="ignore")

print("✅ X SHAPE:", X.shape)

if X.shape[1] == 0:
    raise ValueError("❌ No valid SNP features found!")

# Target
y = data["PH"]   # change to "EH" if needed

# =========================
# 4. NORMALIZATION
# =========================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# 5. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. CONVERT TO TENSORS
# =========================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# =========================
# 7. DEFINE MODEL (DAWL BASE)
# =========================

class DAWLModel(nn.Module):
    def __init__(self, input_dim):
        super(DAWLModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = DAWLModel(X_train.shape[1])

# =========================
# 8. LOSS + OPTIMIZER
# =========================

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 9. TRAIN LOOP
# =========================

epochs = 50

for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"🔥 Epoch {epoch}, Loss: {loss.item():.4f}")

# =========================
# 10. EVALUATION
# =========================

model.eval()
with torch.no_grad():
    preds = model(X_test)
    mse = criterion(preds, y_test)

print("\n✅ Test MSE:", mse.item())

# =========================
# 11. SAVE MODEL
# =========================

SAVE_PATH = os.path.join(BASE_DIR, "dawl_model.pth")

torch.save(model.state_dict(), SAVE_PATH)

print("\n💾 Model saved at:", SAVE_PATH)