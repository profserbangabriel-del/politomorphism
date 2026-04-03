"""
loocv_srm.py — Leave-One-Out Cross-Validation for SRM log-log regression
Politomorphism Research Project | Serban Gabriel Florin
License: CC BY 4.0

Validates: log(SRM) = beta0 + beta1*log(lambda) + beta2*log(V*A*N)

For each symbol i:
  1. Train regression on all symbols EXCEPT i
  2. Predict log(SRM_i) using trained model
  3. Compare predicted vs actual

Output: LOOCV_results.json + LOOCV_report.txt
"""

import json
import math
import numpy as np
from scipy import stats


# ── DATASET — update SRM values as new V_A_N_*.json files arrive ─────────────
# Format: (symbol, lambda, V, A, N, D, SRM_real)
# SRM_real = from V_A_N_*.json files (pipeline output)
# Use None for symbols not yet rerun — they will be excluded from LOOCV

DATASET = [
    # symbol,          lam,    V,      A,      N,      D,      SRM_real
    ("Modi2014",       2.37,   0.1319, 0.2209, 0.1439, 0.7470, 0.000714),
    ("Trump",          7.01,   0.7855, 0.2224, 0.5592, 0.7365, 0.000559),
    ("Putin",          4.90,   0.5025, 0.1082, 0.4024, 0.6502, 0.000904),
    ("Netanyahu",      7.02,   0.4282, 0.1441, 0.8251, 0.6629, 0.000485),
    # Add below as ZIPs arrive:
    # ("Modi2019",     6.33,   V,      A,      N,      D,      SRM),
    # ("Modi2024",     9.11,   V,      A,      N,      D,      SRM),
    # ("Sunflower",    2.00,   V,      A,      N,      D,      SRM),
    # ("Bolsonaro",    10.43,  V,      A,      N,      D,      SRM),
    # ("CharlieHebdo", 104.66, V,      A,      N,      D,      SRM),
    # ("Chavez",       16.67,  V,      A,      N,      D,      SRM),
    # ("Ciolacu",      6.57,   V,      A,      N,      D,      SRM),
    # ("Macron",       12.53,  V,      A,      N,      D,      SRM),
    # ("IranIsrael",   17.81,  V,      A,      N,      D,      SRM),
    # ("Georgescu",    65.33,  V,      A,      N,      D,      SRM),
    # ("Simion",       12.41,  V,      A,      N,      D,      SRM),
    # ("Zelensky",     5.11,   V,      A,      N,      D,      SRM),
    # ("Orban",        2.31,   V,      A,      N,      D,      SRM),
    # ("ChavezAcute",  16.67,  V,      A,      N,      D,      SRM),
]

# Filter out None SRM values
DATASET = [d for d in DATASET if d[6] is not None and d[6] > 0]


def build_features(data):
    """Build log-transformed feature matrix."""
    X = []
    y = []
    names = []
    for sym, lam, V, A, N, D, SRM in data:
        if SRM <= 0 or lam <= 0 or V * A * N <= 0:
            continue
        log_lam = math.log(lam)
        log_VAN = math.log(V * A * N)
        log_SRM = math.log(SRM)
        X.append([1.0, log_lam, log_VAN])
        y.append(log_SRM)
        names.append(sym)
    return np.array(X), np.array(y), names


def ols_fit(X, y):
    """OLS regression: returns coefficients."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def run_full_regression(X, y, names):
    """Full dataset regression with statistics."""
    coeffs = ols_fit(X, y)
    y_pred = X @ coeffs
    residuals = y - y_pred
    n, k = len(y), X.shape[1]

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)

    se = np.sqrt(ss_res / (n - k))
    var_coeff = np.linalg.inv(X.T @ X) * (ss_res / (n - k))
    se_coeffs = np.sqrt(np.diag(var_coeff))
    t_stats = coeffs / se_coeffs
    p_values = [2 * (1 - stats.t.cdf(abs(t), df=n - k)) for t in t_stats]

    return {
        "coefficients": {
            "intercept": round(float(coeffs[0]), 4),
            "log_lambda": round(float(coeffs[1]), 4),
            "log_VAN":    round(float(coeffs[2]), 4),
        },
        "se": {
            "intercept": round(float(se_coeffs[0]), 4),
            "log_lambda": round(float(se_coeffs[1]), 4),
            "log_VAN":    round(float(se_coeffs[2]), 4),
        },
        "t_stats": {
            "intercept": round(float(t_stats[0]), 3),
            "log_lambda": round(float(t_stats[1]), 3),
            "log_VAN":    round(float(t_stats[2]), 3),
        },
        "p_values": {
            "intercept": round(float(p_values[0]), 4),
            "log_lambda": round(float(p_values[1]), 4),
            "log_VAN":    round(float(p_values[2]), 4),
        },
        "R2":     round(float(r2), 4),
        "R2_adj": round(float(r2_adj), 4),
        "RMSE":   round(float(se), 4),
        "n":      n,
    }


def run_loocv(X, y, names):
    """Leave-One-Out Cross-Validation."""
    n = len(y)
    predictions = []
    errors = []

    for i in range(n):
        # Train on all except i
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)

        if len(y_train) < 3:
            continue

        coeffs = ols_fit(X_train, y_train)
        y_pred_i = float(X[i] @ coeffs)
        error = y[i] - y_pred_i

        predictions.append({
            "symbol":        names[i],
            "log_SRM_actual":  round(float(y[i]), 4),
            "log_SRM_pred":    round(y_pred_i, 4),
            "SRM_actual":      round(math.exp(y[i]), 6),
            "SRM_pred":        round(math.exp(y_pred_i), 6),
            "error":           round(float(error), 4),
            "abs_error":       round(abs(float(error)), 4),
            "pct_error":       round(abs(math.exp(y_pred_i) - math.exp(y[i])) / math.exp(y[i]) * 100, 1),
        })
        errors.append(error)

    errors = np.array(errors)
    rmse_loocv = round(float(np.sqrt(np.mean(errors ** 2))), 4)
    mae_loocv  = round(float(np.mean(np.abs(errors))), 4)

    # LOOCV R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_loocv = round(float(1 - ss_res / ss_tot), 4)

    return {
        "predictions":  predictions,
        "RMSE_loocv":   rmse_loocv,
        "MAE_loocv":    mae_loocv,
        "R2_loocv":     r2_loocv,
        "n_folds":      len(predictions),
    }


def print_report(full, loocv):
    """Print human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("SRM LOG-LOG REGRESSION — LOOCV VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("FULL DATASET REGRESSION")
    lines.append(f"  n = {full['n']}")
    lines.append(f"  log(SRM) = {full['coefficients']['intercept']} "
                 f"+ {full['coefficients']['log_lambda']}*log(λ) "
                 f"+ {full['coefficients']['log_VAN']}*log(V·A·N)")
    lines.append(f"  R² = {full['R2']}  |  R²_adj = {full['R2_adj']}")
    lines.append(f"  RMSE = {full['RMSE']}")
    lines.append("")
    lines.append("  Coefficients:")
    for param in ['intercept', 'log_lambda', 'log_VAN']:
        c = full['coefficients'][param]
        se = full['se'][param]
        t = full['t_stats'][param]
        p = full['p_values'][param]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.10 else 'n.s.'
        lines.append(f"    {param:<15} β={c:>8.4f}  SE={se:.4f}  t={t:>7.3f}  p={p:.4f} {sig}")
    lines.append("")

    lines.append("LEAVE-ONE-OUT CROSS-VALIDATION")
    lines.append(f"  n folds = {loocv['n_folds']}")
    lines.append(f"  RMSE_loocv = {loocv['RMSE_loocv']}")
    lines.append(f"  MAE_loocv  = {loocv['MAE_loocv']}")
    lines.append(f"  R²_loocv   = {loocv['R2_loocv']}")
    lines.append("")
    lines.append(f"  {'Symbol':<18} {'SRM_actual':>12} {'SRM_pred':>12} {'%_error':>10}")
    lines.append("  " + "-" * 55)
    for p in loocv['predictions']:
        lines.append(f"  {p['symbol']:<18} {p['SRM_actual']:>12.6f} {p['SRM_pred']:>12.6f} {p['pct_error']:>9.1f}%")
    lines.append("")

    # Interpretation
    r2_full = full['R2_adj']
    r2_cv = loocv['R2_loocv']
    drop = round(r2_full - r2_cv, 4)
    lines.append("INTERPRETATION")
    lines.append(f"  R²_adj (full) = {r2_full}  →  R²_loocv = {r2_cv}")
    lines.append(f"  R² drop from LOOCV = {drop}")
    if drop < 0.10:
        lines.append("  ✓ Stable: model generalizes well (drop < 0.10)")
    elif drop < 0.20:
        lines.append("  ⚠ Moderate overfitting (drop 0.10-0.20) — report as limitation")
    else:
        lines.append("  ✗ Overfitting detected (drop > 0.20) — revise model")
    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report


if __name__ == "__main__":
    print(f"\n=== LOOCV SRM Regression — n={len(DATASET)} symbols ===\n")

    if len(DATASET) < 4:
        print(f"WARNING: Only {len(DATASET)} symbols with real SRM values.")
        print("LOOCV requires at least 4. Add more symbols to DATASET.")
        print("Run more V_A_N_*.json pipeline jobs and update DATASET above.")
    else:
        X, y, names = build_features(DATASET)

        print("Building full regression...")
        full = run_full_regression(X, y, names)

        print("Running LOOCV...")
        loocv = run_loocv(X, y, names)

        report = print_report(full, loocv)

        output = {
            "n_symbols": len(DATASET),
            "symbols": [d[0] for d in DATASET],
            "full_regression": full,
            "loocv": loocv,
            "note": "Update DATASET with V_A_N_*.json values as pipeline runs complete"
        }

        with open("LOOCV_results.json", "w") as f:
            json.dump(output, f, indent=2)
        with open("LOOCV_report.txt", "w") as f:
            f.write(report)

        print("\nSaved: LOOCV_results.json + LOOCV_report.txt")
