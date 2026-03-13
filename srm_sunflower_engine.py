# srm_sunflower_engine.py
import json
import math

# Values from your paper (you can later load from data_sunflower/ if you have CSV/JSON)
V = 0.750          # Viral Velocity
A = 0.393          # Affective Weight
D = 0.7737         # Semantic Drift
N = 0.600          # Network Coverage
lambda_d = 2.0     # Penalty constant

semantic_factor = math.exp(-lambda_d * D)
SRM = V * A * semantic_factor * N

result = {
    "symbol": "Sunflower Movement",
    "period": "2014 (main protests) + international tail",
    "V": V,
    "A": A,
    "D": D,
    "semantic_factor": round(semantic_factor, 4),
    "N": N,
    "SRM": round(SRM, 4),
    "resonance_level": "LOW RESONANCE" if SRM < 0.05 else "MEDIUM RESONANCE" if SRM < 0.1 else "HIGH RESONANCE",
    "countries": ["Taiwan", "United States", "United Kingdom", "Japan", "Hong Kong", "Australia", "France", "Germany", "Canada", "South Korea", "Singapore", "Netherlands"],
    "note": "Resonance Paradox: high V and N, but strong penalty from high D"
}

# Save result
with open("SRM_sunflower_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"Calculated SRM: {SRM:.4f} → {result['resonance_level']}")

# Optional: add matplotlib code here later to generate SRM_sunflower_grafic.png (like other cases)
