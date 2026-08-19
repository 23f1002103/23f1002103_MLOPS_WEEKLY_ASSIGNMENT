import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Load original training data (before location column, just the raw features)
df = pd.read_csv("data/v1data.csv")
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

X_original = df[features].copy()

# Simulate "production" data: shift petal_length up by 1.5 (pretend real-world data changed)
X_production = X_original.copy()
X_production['petal_length'] = X_production['petal_length'] + 1.5

# Run KS test (Kolmogorov-Smirnov) for each feature: original vs "production"
print("--- Drift Detection Results (KS Test) ---")
for col in features:
    stat, p_value = stats.ks_2samp(X_original[col], X_production[col])
    drift_flag = "DRIFT DETECTED" if p_value < 0.05 else "No significant drift"
    print(f"{col}: KS statistic={stat:.4f}, p-value={p_value:.6f} -> {drift_flag}")

# Plot histograms: original vs production, for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(features):
    axes[i].hist(X_original[col], bins=15, alpha=0.5, label='Original (training)', color='blue')
    axes[i].hist(X_production[col], bins=15, alpha=0.5, label='Production (simulated)', color='red')
    axes[i].set_title(col)
    axes[i].legend()

plt.tight_layout()
plt.savefig("drift_comparison.png")
print("\nSaved drift_comparison.png")