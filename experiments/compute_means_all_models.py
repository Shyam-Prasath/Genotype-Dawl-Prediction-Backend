import os
import pandas as pd

# ===========================
# Setup paths
# ===========================

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

file_path = os.path.join(project_root, "cross_pop_results_all_models.csv")

# ===========================
# Load data
# ===========================

df = pd.read_csv(file_path)

print("Loaded rows:", len(df))

# ===========================
# Compute mean ± std
# ===========================

summary = (
    df.groupby(["model", "source_panel", "target_panel"])["correlation"]
    .agg(["mean", "std"])
    .reset_index()
)

# Round values
summary["mean"] = summary["mean"].round(3)
summary["std"] = summary["std"].round(3)

# Combine mean ± std
summary["result"] = summary["mean"].astype(str) + " ± " + summary["std"].astype(str)

# ===========================
# Pivot for table format
# ===========================

table = summary.pivot_table(
    index=["source_panel", "target_panel"],
    columns="model",
    values="result",
    aggfunc="first"
).reset_index()

# ===========================
# Save
# ===========================

output_path = os.path.join(project_root, "final_crosspop_table.csv")
table.to_csv(output_path, index=False)

print("\nFinal Table saved at:", output_path)
print("\nPreview:\n")
print(table)