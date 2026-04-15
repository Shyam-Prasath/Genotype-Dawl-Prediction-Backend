import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# =========================
# 1. CREATE SYNTHETIC DATA
# =========================

np.random.seed(42)

data = []

for _ in range(2000):
    ph = np.random.uniform(50, 250)
    eh = np.random.uniform(10, ph)

    ratio = eh / ph

    # Targets (still logic, but used to TRAIN ML)
    vigor = (ph + eh) / 2
    stability = (1 - abs(0.5 - ratio)) * 100

    if ratio > 0.5:
        cls = 2  # Good
    elif ratio > 0.3:
        cls = 1  # Moderate
    else:
        cls = 0  # Poor

    data.append([ph, eh, vigor, stability, cls])

df = pd.DataFrame(data, columns=["PH", "EH", "Vigor", "Stability", "Class"])

# =========================
# 2. PREPARE DATA
# =========================

X = df[["PH", "EH"]].values
y = df[["Vigor", "Stability"]].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# =========================
# 3. MODEL
# =========================

class AnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # vigor + stability
        )

    def forward(self, x):
        return self.net(x)

model = AnalysisModel()

# =========================
# 4. TRAIN
# =========================

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(200):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print("Loss:", loss.item())

# =========================
# 5. SAVE
# =========================

torch.save(model.state_dict(), "analysis_model.pth")

print("✅ analysis_model.pth saved")