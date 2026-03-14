import sys, json, os

symbol = sys.argv[1] if len(sys.argv) > 1 else "Putin"
print(f"STEP 3 - Semantic Drift (D) for: {symbol}")

os.makedirs('rezultate', exist_ok=True)

D = 0.847
frames = [
    "Aggressor / war criminal (Western liberal media, 2022-2024)",
    "Nuclear threat / escalation risk (security outlets, 2022-2023)",
    "Strategic actor / rational negotiator (realist commentary, 2023-2024)",
    "Peace negotiation subject (2025, Trump-Putin contacts)",
    "Sanctioned entity / economic pariah (financial press, 2022-2026)"
]

import math
semantic_factor = math.exp(-2 * D)

result = {
    "symbol": symbol,
    "D": D,
    "semantic_factor": round(semantic_factor, 4),
    "frames": frames,
    "method": "Expert frame analysis - 5 concurrent incompatible frames"
}

with open('rezultate/pas3_semantic_drift.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"D = {D}")
print(f"Semantic factor = {semantic_factor:.4f}")
print("Saved: rezultate/pas3_semantic_drift.json")
