import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from tqdm import tqdm

# ===========================
# Setup paths
# ===========================

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

print("Running Random Forest Cross-Population Evaluation")

# ===========================
# Load Dataset
# ===========================

data_path = os.path.join(project_root, "final_merged_dataset.txt")
df = pd.read_csv(data_path, sep="\t")

panels = df["Panel"].unique()
print("Available Panels:", panels)

# Features and labels
X_all = df.drop(columns=["Base_ID", "Full_name", "Panel", "PH", "EH"]).values
y_ph_all = df["PH"].values
panel_labels = df["Panel"].values

# ===========================
# Global Standardization + PCA
# ===========================

scaler_global = StandardScaler()
X_scaled_global = scaler_global.fit_transform(X_all)

pca = PCA(n_components=20)
X_pca_global = pca.fit_transform(X_scaled_global)

# ===========================
# Cross-Population Evaluation
# ===========================

seeds = [42, 123, 999, 2024, 7]

panel_results = []
degradation_results = []

total_iterations = len(panels) * len(seeds)

with tqdm(total=total_iterations, desc="Running Experiments") as pbar:

    for target_panel in panels:

        print("\n==============================")
        print(f"Train: ALL_EXCEPT_{target_panel} → Test: {target_panel}")
        print("==============================")

        train_mask = panel_labels != target_panel
        test_mask = panel_labels == target_panel

        X_train = X_all[train_mask]
        y_train = y_ph_all[train_mask]

        X_test = X_all[test_mask]
        y_test = y_ph_all[test_mask]

        # PCA subsets
        X_train_pca = X_pca_global[train_mask]
        X_test_pca = X_pca_global[test_mask]

        train_centroid = np.mean(X_train_pca, axis=0)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for seed in seeds:

            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=seed,
                n_jobs=-1
            )

            rf.fit(X_train_scaled, y_train)

            preds = rf.predict(X_test_scaled)

            corr, _ = pearsonr(y_test, preds)

            panel_results.append({
                "model": "RandomForest",
                "source_panel": "ALL_EXCEPT_" + target_panel,
                "target_panel": target_panel,
                "seed": seed,
                "correlation": corr
            })

            # Genotype-level degradation
            abs_errors = np.abs(y_test - preds)

            for i in range(len(X_test_pca)):
                distance = np.linalg.norm(X_test_pca[i] - train_centroid)

                degradation_results.append({
                    "model": "RandomForest",
                    "source_panel": "ALL_EXCEPT_" + target_panel,
                    "target_panel": target_panel,
                    "distance": distance,
                    "abs_error": abs_errors[i]
                })

            pbar.update(1)

# ===========================
# Save Results
# ===========================

panel_df = pd.DataFrame(panel_results)
panel_df.to_csv(os.path.join(project_root, "cross_pop_results_rf.csv"), index=False)

degradation_df = pd.DataFrame(degradation_results)
degradation_df.to_csv(
    os.path.join(project_root, "genotype_degradation_results_rf.csv"),
    index=False
)

print("\nSaved Files:")
print(" - cross_pop_results_rf.csv")
print(" - genotype_degradation_results_rf.csv")
print("Total genotype rows:", len(degradation_df))