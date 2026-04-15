import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===========================
# Known Values From Experiments
# ===========================

# Original USP genetic distance
original_distance = 70.0

# Augmentation ratios
ratios = np.array([0.0, 0.05, 0.10, 0.20, 0.30])

# Observed PH performance from augmentation experiment
performance = np.array([
    -0.1877,
    0.0306,
    0.1015,
    0.2461,
    0.4020
])

# ===========================
# Compute Effective Distance
# ===========================

effective_distance = (1 - ratios) * original_distance
effective_distance = effective_distance.reshape(-1, 1)

# ===========================
# Fit Regression Model
# ===========================

model = LinearRegression()
model.fit(effective_distance, performance)

predicted = model.predict(effective_distance)
r2 = r2_score(performance, predicted)

print("Continuous Domain-Shift Degradation Model")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R2 Score:", round(r2, 4))

# ===========================
# Example Prediction
# ===========================

test_distance = np.array([[40]])
expected_perf = model.predict(test_distance)[0]

print("\nIf effective distance = 40")
print("Expected PH performance ≈", round(expected_perf, 4))