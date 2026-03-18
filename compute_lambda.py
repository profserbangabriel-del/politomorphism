import pandas as pd
import numpy as np
from scipy.optimize import brentq

# Load the CSV file
df = pd.read_csv('chariel hebdo counts.csv', parse_dates=['date'])

# Use the 'count' column
counts = df['count'].values

# Calculate average and peak
avg_count = np.mean(counts)
peak_count = np.max(counts)
ratio = avg_count / peak_count

# Calculate period in years
start_date = df['date'].min()
end_date = df['date'].max()
T_years = (end_date - start_date).days / 365.25

# Function to solve for lambda
def func(lam):
    if lam == 0:
        return 1 - ratio
    return (1 - np.exp(-lam * T_years)) / (lam * T_years) - ratio

# Find a bracket where f(a) and f(b) have opposite signs
a = 1e-6          # lower bound (near zero)
b = 100           # initial upper bound
max_iter = 10     # safety to avoid infinite loop
for _ in range(max_iter):
    if func(a) * func(b) <= 0:
        break
    b *= 10        # increase upper bound
else:
    raise ValueError("Could not find a bracket with sign change. Check ratio or T.")

# Solve numerically for lambda
lambda_empirical = brentq(func, a, b)

# Print results
print(f"Period: {start_date.date()} – {end_date.date()} ({T_years:.3f} years)")
print(f"Average mentions: {avg_count:.2f}")
print(f"Peak mentions: {peak_count}")
print(f"Average/Peak ratio: {ratio:.6f}")
print(f"Empirical λ = {lambda_empirical:.4f}")
