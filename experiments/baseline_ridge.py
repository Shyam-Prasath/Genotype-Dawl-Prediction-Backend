from sklearn.linear_model import Ridge
from utils import load_data, evaluate_model

# Load data
X, y_ph, y_eh = load_data()

# Model
model = Ridge(alpha=1.0)

# Evaluate PH
corr_ph, rmse_ph = evaluate_model(model, X, y_ph)
print("Ridge Regression - PH")
print("Correlation:", corr_ph)
print("RMSE:", rmse_ph)

# Evaluate EH
corr_eh, rmse_eh = evaluate_model(model, X, y_eh)
print("\nRidge Regression - EH")
print("Correlation:", corr_eh)
print("RMSE:", rmse_eh)