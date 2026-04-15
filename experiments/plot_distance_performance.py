import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ===========================
# Data
# ===========================

distances = np.array([10.9236, 10.9236, 70.0])
performance = np.array([0.8581, 0.6193, -0.0818])
labels = ["ASSO", "NCRIPS", "USP"]

# Regression
model = LinearRegression()
model.fit(distances.reshape(-1,1), performance)
r2 = model.score(distances.reshape(-1,1), performance)

x_line = np.linspace(distances.min(), distances.max(), 100)
y_line = model.predict(x_line.reshape(-1,1))

# ===========================
# Plot
# ===========================

fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(distances, performance, s=120)

for i in range(len(labels)):
    ax.annotate(labels[i],
                (distances[i], performance[i]),
                xytext=(5,5),
                textcoords="offset points")

ax.plot(x_line, y_line, linewidth=2)

ax.set_xlabel("Genetic Distance (PCA Space)")
ax.set_ylabel("Cross-Population PH Correlation")
ax.set_title("Performance Degradation as a Function of Genetic Distance")

# Place R2 safely inside plot
ax.text(0.05, 0.90, f"$R^2$ = {r2:.3f}",
        transform=ax.transAxes)

ax.grid(alpha=0.3)

# ❌ REMOVE tight_layout()
# plt.tight_layout()

plt.savefig("Figure3_Distance_vs_Performance.png", dpi=300)
plt.show()

print("Figure3_Distance_vs_Performance.png saved.")