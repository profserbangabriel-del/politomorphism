import pandas as pd
import numpy as np
from scipy.optimize import brentq

df = pd.read_csv('chariel hebdo counts.csv', parse_dates=['date'])
counts = df['count'].values

avg_count = np.mean(counts)
peak_count = np.max(counts)
raport = avg_count / peak_count

start_date = df['date'].min()
end_date = df['date'].max()
T_ani = (end_date - start_date).days / 365.25

def functie(lam):
    if lam == 0:
        return 1 - raport
    return (1 - np.exp(-lam * T_ani)) / (lam * T_ani) - raport

lambda_empiric = brentq(functie, 1e-6, 100)

print(f"Perioada: {start_date.date()} – {end_date.date()} ({T_ani:.3f} ani)")
print(f"Media mențiunilor: {avg_count:.2f}")
print(f"Vârful mențiunilor: {peak_count}")
print(f"Raport medie/vârf: {raport:.6f}")
print(f"λ empiric = {lambda_empiric:.4f}")
