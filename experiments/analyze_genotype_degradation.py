import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm

# ===========================
# Setup
# ===========================

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

data_path = os.path.join(project_root, "genotype_degradation_results.csv")

df = pd.read_csv(data_path)

print("Total raw rows:", len(df))

# Normalize column names
df.columns = df.columns.str.lower()

print("Columns found:", df.columns.tolist())

# ===========================
# 1️⃣ Remove Pseudo-Replication
# ===========================

# Correct grouping based on YOUR CSV:
# (model, source_panel, target_panel, distance)

df_grouped = (
    df.groupby(
        ["model", "source_panel", "target_panel", "distance"],
        as_index=False
    )
    .agg(mean_abs_error=("abs_error", "mean"))
)

print("Unique biological samples:", len(df_grouped))

# ===========================
# 2️⃣ Select Model to Analyze
# ===========================

model_name = "Ridge"   # Change to "MLP" if needed

df_model = df_grouped[df_grouped["model"] == model_name]

distances = df_model["distance"].values
errors = df_model["mean_abs_error"].values

print(f"\nAnalyzing Model: {model_name}")
print("Total genotypes analyzed:", len(distances))

# ===========================
# 3️⃣ Pearson Correlation
# ===========================

r_value, p_value = pearsonr(distances, errors)

print("\nGenotype-Level Degradation Analysis")
print("Pearson r:", round(r_value, 4))
print("p-value:", p_value)

# ===========================
# 4️⃣ Linear Regression + 95% CI
# ===========================

X = sm.add_constant(distances)
ols_model = sm.OLS(errors, X).fit()

pred = ols_model.get_prediction(X)
pred_summary = pred.summary_frame(alpha=0.05)

# Sort for clean plotting
sorted_idx = np.argsort(distances)
distances_sorted = distances[sorted_idx]
mean_pred_sorted = pred_summary["mean"].values[sorted_idx]
ci_lower_sorted = pred_summary["mean_ci_lower"].values[sorted_idx]
ci_upper_sorted = pred_summary["mean_ci_upper"].values[sorted_idx]

# ===========================
# 5️⃣ Plot (IEEE-ready)
# ===========================

plt.figure(figsize=(8, 6))

plt.scatter(distances, errors, alpha=0.3, label="Genotypes")
plt.plot(distances_sorted, mean_pred_sorted, label="Linear Fit")
plt.fill_between(
    distances_sorted,
    ci_lower_sorted,
    ci_upper_sorted,
    alpha=0.2,
    label="95% CI"
)

plt.xlabel("Genetic Distance to Training Centroid (PCA Space)")
plt.ylabel("Mean Absolute Prediction Error")
plt.title(f"Genotype-Level Degradation ({model_name})")

plt.text(
    0.05,
    0.90,
    f"r = {r_value:.3f}\np = {p_value:.2e}",
    transform=plt.gca().transAxes
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output_path = os.path.join(
    project_root,
    f"Figure3_Genotype_Degradation_{model_name}.png"
)

plt.savefig(output_path, dpi=300)
plt.show()

print("\nSaved:", output_path)