import numpy as np
from scipy.stats import pearsonr

# ===========================
# Manually Insert Values
# ===========================

# Genetic distances (from PCA analysis)
distance = {
    "ASSO": 10.9236,      # distance from training panels
    "NCRIPS": 10.9236,    # ASSO-NCRIPS symmetric small shift
    "USP": 70.0           # approximate average of 67 and 72
}

# Cross-population performance (Ridge PH)
performance_ph = {
    "ASSO": 0.8581,
    "NCRIPS": 0.6193,
    "USP": -0.0818
}

# Cross-population performance (Ridge EH)
performance_eh = {
    "ASSO": 0.8765,
    "NCRIPS": 0.6857,
    "USP": 0.0271
}

# ===========================
# Convert to Arrays
# ===========================

dist_vals = np.array([distance[k] for k in ["ASSO", "NCRIPS", "USP"]])
perf_ph_vals = np.array([performance_ph[k] for k in ["ASSO", "NCRIPS", "USP"]])
perf_eh_vals = np.array([performance_eh[k] for k in ["ASSO", "NCRIPS", "USP"]])

# ===========================
# Correlation Analysis
# ===========================

corr_ph, _ = pearsonr(dist_vals, perf_ph_vals)
corr_eh, _ = pearsonr(dist_vals, perf_eh_vals)

print("Distance vs PH Performance Correlation:", round(corr_ph,4))
print("Distance vs EH Performance Correlation:", round(corr_eh,4))