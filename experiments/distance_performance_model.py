import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===========================
# Data From Your Experiments
# ===========================

# Distances (from PCA)
distances = np.array([
    10.9236,   # ASSO
    10.9236,   # NCRIPS
    70.0       # USP
]).reshape(-1,1)

# Ridge PH cross-pop performance
ph_performance = np.array([
    0.8581,
    0.6193,
    -0.0818
])

# ===========================
# Fit Regression
# ===========================

model = LinearRegression()
model.fit(distances, ph_performance)

predicted = model.predict(distances)

r2 = r2_score(ph_performance, predicted)

print("Distance → PH Performance Model")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R2 Score:", round(r2,4))

# Example prediction
test_distance = np.array([[50]])
expected_perf = model.predict(test_distance)[0]

print("\nIf distance = 50 → Expected PH Performance ≈",
      round(expected_perf,4))