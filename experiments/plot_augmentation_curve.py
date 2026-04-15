import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
ratios = np.array([0, 5, 10, 20, 30])
performance = np.array([-0.1877, 0.0306, 0.1015, 0.2461, 0.4020])

# Regression
model = LinearRegression()
model.fit(ratios.reshape(-1,1), performance)

x_line = np.linspace(0, 30, 100).reshape(-1,1)
y_line = model.predict(x_line)

# Plot
plt.figure(figsize=(8,6))

plt.plot(ratios, performance, marker='o', linewidth=2)
plt.plot(x_line, y_line, linewidth=2)

for i, val in enumerate(performance):
    plt.text(ratios[i], performance[i] + 0.02, f"{val:.2f}", ha='center')

plt.xlabel("Target Population Added (%)")
plt.ylabel("Cross-Population PH Correlation")
plt.title("Performance Recovery via Target Augmentation")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("Figure4_Augmentation_Recovery.png", dpi=300)
plt.show()

print("Figure4_Augmentation_Recovery.png saved.")