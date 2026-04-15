import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("final_crosspop_table_v2.csv")

# Create transfer labels
df["transfer"] = df["source_panel"] + "→" + df["target_panel"]

# Extract mean values (remove ± std)
def extract_mean(val):
    return float(str(val).split("±")[0].strip())

df["MLP"] = df["MLP"].apply(extract_mean)
df["MLP_DAWL"] = df["MLP_DAWL"].apply(extract_mean)
df["Ridge"] = df["Ridge"].apply(extract_mean)
df["Ridge_DAWL"] = df["Ridge_DAWL"].apply(extract_mean)

# Sort for consistent ordering
df = df.sort_values(by=["source_panel", "target_panel"])

labels = df["transfer"].values
x = np.arange(len(labels))
width = 0.2

# Reset style (IMPORTANT)
plt.style.use('default')

fig, ax = plt.subplots(figsize=(10,6))

# White background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot bars
ax.bar(x - 1.5*width, df["MLP"], width, label='MLP')
ax.bar(x - 0.5*width, df["MLP_DAWL"], width, label='MLP + DAWL')
ax.bar(x + 0.5*width, df["Ridge"], width, label='Ridge')
ax.bar(x + 1.5*width, df["Ridge_DAWL"], width, label='Ridge + DAWL')

# Labels
ax.set_xlabel("Source → Target Panel")
ax.set_ylabel("Pearson Correlation (r)")

# X ticks
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha='right')

# Y-axis limits
ax.set_ylim(-0.4, 0.9)

# Light grid
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Legend outside (clean)
ax.legend(loc='upper left', bbox_to_anchor=(1,1), frameon=False)

# Tight layout
plt.tight_layout()

# Save high-quality image
plt.savefig("Figure_DAWL_Comparison_FINAL.png", dpi=300, bbox_inches='tight')

plt.show()