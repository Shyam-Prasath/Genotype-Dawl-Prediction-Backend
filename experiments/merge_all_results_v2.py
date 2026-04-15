import os
import pandas as pd

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

file_base = os.path.join(project_root, "cross_pop_results.csv")
file_rf = os.path.join(project_root, "cross_pop_results_rf_fixed.csv")
file_dawl = os.path.join(project_root, "cross_pop_results_dawl.csv")

# Load
df_base = pd.read_csv(file_base)
df_rf = pd.read_csv(file_rf)
df_dawl = pd.read_csv(file_dawl)

# Merge
df_all = pd.concat([df_base, df_rf, df_dawl], ignore_index=True)

# Save
output_path = os.path.join(project_root, "cross_pop_results_all_models_v2.csv")
df_all.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Models:", df_all["model"].unique())