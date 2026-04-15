import os
import pandas as pd

# ===========================
# Setup
# ===========================

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file_path = os.path.join(project_root, "cross_pop_results_all_models_v2.csv")

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

summary["mean"] = summary["mean"].round(3)
summary["std"] = summary["std"].round(3)

summary["result"] = summary["mean"].astype(str) + " ± " + summary["std"].astype(str)

# ===========================
# Pivot
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

output_path = os.path.join(project_root, "final_crosspop_table_v2.csv")
table.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("\nPreview:\n")
print(table)