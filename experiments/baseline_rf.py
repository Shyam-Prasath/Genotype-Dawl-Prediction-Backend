import sys
import os

# Make sure we can import utils
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from sklearn.ensemble import RandomForestRegressor
from experiments.utils import load_data, evaluate_model


# =====================================
# Load Data
# =====================================
X, y_ph, y_eh = load_data()


# =====================================
# Random Forest Model
# =====================================
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


# =====================================
# Evaluate PH
# =====================================
corr_ph, rmse_ph = evaluate_model(model, X, y_ph)

print("Random Forest - PH")
print("Correlation:", corr_ph)
print("RMSE:", rmse_ph)


# =====================================
# Evaluate EH
# =====================================
corr_eh, rmse_eh = evaluate_model(model, X, y_eh)

print("\nRandom Forest - EH")
print("Correlation:", corr_eh)
print("RMSE:", rmse_eh)