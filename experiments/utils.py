import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm

def load_data():
    # Get absolute path to project root
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    file_path = os.path.join(project_root, "final_merged_dataset.txt")

    df = pd.read_csv(file_path, sep="\t")

    X = df.drop(columns=["Base_ID", "Full_name", "Panel", "PH", "EH"])
    y_ph = df["PH"].values
    y_eh = df["EH"].values

    return X.values, y_ph, y_eh


def evaluate_model(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    correlations = []
    rmses = []

    print("\nStarting Cross Validation...\n")

    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(X), total=n_splits)):

        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        corr, _ = pearsonr(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print(f"Fold {fold+1} Correlation: {corr:.4f}")
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

        correlations.append(corr)
        rmses.append(rmse)

    print("\nCross Validation Completed.\n")

    return np.mean(correlations), np.mean(rmses)