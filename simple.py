import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv("augmentation_results_usp.csv")
df.columns = df.columns.str.lower()

summary = (
    df.groupby("augmentation_pct")
      .agg(
          ridge_mean=("ridge_corr", "mean"),
          ridge_std=("ridge_corr", "std"),
          mlp_mean=("mlp_corr", "mean"),
          mlp_std=("mlp_corr", "std")
      )
      .reset_index()
      .round(3)
)

print(summary)