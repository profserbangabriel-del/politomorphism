import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# 1. Load Trump data
df = pd.read_csv('TRUMP+DATA.csv', quotechar='"', skipinitialspace=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 2. SRM Variables (Validation Case 4)
V, A, D, N = 0.90, 0.72, 0.25, 0.95

# 3. Formula: SRM = V * A * exp(-2 * D) * N
srm_score = V * A * np.exp(-2 * D) * N

# 4. Export results
with open('trump_srm_report.json', 'w') as f:
    json.dump({"symbol": "Donald Trump", "srm_score": round(srm_score, 4)}, f, indent=4)

# 5. Export chart
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['ratio'], color='red', label='Trump Visibility')
plt.title('SRM Validation: Donald Trump (English Report)')
plt.savefig('trump_srm_chart.png')
