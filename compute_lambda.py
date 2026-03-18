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

# Solve numerically for lambda
lambda_empirical = brentq(func, 1e-6, 100)

# Print results
print(f"Period: {start_date.date()} – {end_date.date()} ({T_years:.3f} years)")
print(f"Average mentions: {avg_count:.2f}")
print(f"Peak mentions: {peak_count}")
print(f"Average/Peak ratio: {ratio:.6f}")
print(f"Empirical λ = {lambda_empirical:.4f}")
