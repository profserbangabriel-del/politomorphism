#!/usr/bin/env python3
"""
eef_loocv.py — Leave-One-Out Cross-Validation for EEF (FIIM v2.1)
Politomorphism Framework | Serban Gabriel Florin
ORCID: 0009-0000-2266-3356 | OSF: 10.17605/OSF.IO/HYDNZ

LOOCV logic:
  For each of the 60 country-year observations:
    1. Remove observation i from the dataset (test set = 1 obs)
    2. Fit a threshold classifier on the remaining 59 observations
       using IS_agg as the sole predictor (zone boundary learned from training set)
    3. Predict zone for observation i
    4. Compare predicted zone vs actual zone
  Report: accuracy, F1 macro, per-class metrics, confusion matrix,
          per-country accuracy, and per-zone breakdown.

Zone encoding:
  MODERATE -> 0
  HIGH     -> 1
  CRITICAL -> 2  (not present in FIIM v2.1 output, but supported)

Threshold learning (from training set):
  - MODERATE/HIGH boundary = mean IS_agg of observations that cross the boundary
    (i.e., average of max IS in MODERATE group + min IS in HIGH group)
  - This is a data-driven threshold, not the fixed 0.55/0.65 hard-coded rule.
  - If only one zone exists in training set, threshold is set to the
    global midpoint of training IS_agg range.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_CSV   = "EEF_FIIM_v21_All.csv"
OUTPUT_CSV  = "EEF_LOOCV_Results.csv"
SUMMARY_CSV = "EEF_LOOCV_Summary.csv"
FEATURE_COL = "IS_agg"   # primary predictor
TARGET_COL  = "zone"     # MODERATE / HIGH / CRITICAL

ZONE_ORDER = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
ZONE_MAP   = {z: i for i, z in enumerate(ZONE_ORDER)}

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df["Delta_IS"]  = df["Delta_IS"].fillna(0.0)
df["delta_FH"]  = df["delta_FH"].fillna(0.0)
df = df.reset_index(drop=True)

n = len(df)
print(f"Loaded {n} observations | Zones: {df[TARGET_COL].value_counts().to_dict()}")

# ── HELPER: learn threshold from training set ────────────────────────────────
def learn_threshold(train_df, lower_zone, upper_zone):
    """
    Learn the IS_agg boundary between lower_zone and upper_zone
    from the training set.
    Returns the midpoint between the max IS of lower_zone and
    min IS of upper_zone in the training data.
    If either zone is absent, returns None.
    """
    lower_vals = train_df.loc[train_df[TARGET_COL] == lower_zone, FEATURE_COL]
    upper_vals = train_df.loc[train_df[TARGET_COL] == upper_zone, FEATURE_COL]
    if lower_vals.empty or upper_vals.empty:
        return None
    return (lower_vals.max() + upper_vals.min()) / 2.0


def predict_zone(is_val, t_mod_high, t_high_crit):
    """
    Predict zone given IS_agg value and learned thresholds.
    t_mod_high  : MODERATE/HIGH boundary
    t_high_crit : HIGH/CRITICAL boundary (may be None)
    """
    if t_high_crit is not None and is_val >= t_high_crit:
        return "CRITICAL"
    if t_mod_high is not None and is_val >= t_mod_high:
        return "HIGH"
    return "MODERATE"


# ── LOOCV LOOP ───────────────────────────────────────────────────────────────
results = []

for i in range(n):
    test_row  = df.iloc[i]
    train_df  = df.drop(index=i).reset_index(drop=True)

    # Learn thresholds from training set
    t_mod_high  = learn_threshold(train_df, "MODERATE", "HIGH")
    t_high_crit = learn_threshold(train_df, "HIGH", "CRITICAL")

    # Fallback: if threshold cannot be learned (only 1 zone in training),
    # use midpoint of training IS_agg range
    if t_mod_high is None:
        t_mod_high = train_df[FEATURE_COL].mean()

    # Predict
    is_val    = test_row[FEATURE_COL]
    pred_zone = predict_zone(is_val, t_mod_high, t_high_crit)
    true_zone = test_row[TARGET_COL]
    correct   = int(pred_zone == true_zone)

    results.append({
        "country"     : test_row["country"],
        "year"        : int(test_row["year"]),
        "IS_agg"      : round(is_val, 4),
        "true_zone"   : true_zone,
        "pred_zone"   : pred_zone,
        "correct"     : correct,
        "t_mod_high"  : round(t_mod_high, 4),
        "t_high_crit" : round(t_high_crit, 4) if t_high_crit else None,
        "electoral_override": bool(test_row.get("electoral_override", False)),
    })

results_df = pd.DataFrame(results)

# ── METRICS ──────────────────────────────────────────────────────────────────
overall_accuracy = results_df["correct"].mean()

# Per-zone metrics (precision, recall, F1)
zones_present = results_df["true_zone"].unique().tolist()

zone_metrics = {}
for zone in zones_present:
    tp = ((results_df["true_zone"] == zone) & (results_df["pred_zone"] == zone)).sum()
    fp = ((results_df["true_zone"] != zone) & (results_df["pred_zone"] == zone)).sum()
    fn = ((results_df["true_zone"] == zone) & (results_df["pred_zone"] != zone)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    support   = (results_df["true_zone"] == zone).sum()
    zone_metrics[zone] = {
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1"       : round(f1,        4),
        "support"  : int(support),
        "tp"       : int(tp),
        "fp"       : int(fp),
        "fn"       : int(fn),
    }

# Macro F1
f1_macro = np.mean([zone_metrics[z]["f1"] for z in zones_present])

# Per-country accuracy
country_acc = (results_df.groupby("country")["correct"]
               .agg(["sum","count"])
               .rename(columns={"sum":"correct","count":"total"}))
country_acc["accuracy"] = (country_acc["correct"] / country_acc["total"]).round(4)

# Confusion matrix
all_zones = sorted(zones_present, key=lambda z: ZONE_MAP.get(z, 99))
conf_matrix = pd.crosstab(
    results_df["true_zone"],
    results_df["pred_zone"],
    rownames=["True"],
    colnames=["Predicted"]
)

# Misclassified rows
misclassified = results_df[results_df["correct"] == 0].copy()

# Threshold stability across folds
threshold_stats = {
    "t_mod_high_mean" : round(results_df["t_mod_high"].mean(), 4),
    "t_mod_high_std"  : round(results_df["t_mod_high"].std(),  4),
    "t_mod_high_min"  : round(results_df["t_mod_high"].min(),  4),
    "t_mod_high_max"  : round(results_df["t_mod_high"].max(),  4),
}

# ── PRINT REPORT ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EEF LOOCV REPORT")
print("="*60)
print(f"\nN observations : {n}")
print(f"Overall accuracy: {overall_accuracy:.4f}  ({overall_accuracy*100:.1f}%)")
print(f"F1 macro        : {f1_macro:.4f}")

print("\n── Per-Zone Metrics ──")
for zone in all_zones:
    m = zone_metrics[zone]
    print(f"  {zone:<10} | P={m['precision']:.3f}  R={m['recall']:.3f}  "
          f"F1={m['f1']:.3f}  support={m['support']}")

print("\n── Per-Country Accuracy ──")
for country, row in country_acc.iterrows():
    print(f"  {country:<10} | {int(row['correct'])}/{int(row['total'])}  "
          f"= {row['accuracy']*100:.1f}%")

print("\n── Confusion Matrix ──")
print(conf_matrix.to_string())

print("\n── Threshold Stability (MODERATE/HIGH boundary across folds) ──")
print(f"  Mean : {threshold_stats['t_mod_high_mean']}")
print(f"  Std  : {threshold_stats['t_mod_high_std']}")
print(f"  Range: [{threshold_stats['t_mod_high_min']}, {threshold_stats['t_mod_high_max']}]")

print("\n── Misclassified Observations ──")
if misclassified.empty:
    print("  None — perfect LOOCV accuracy.")
else:
    for _, row in misclassified.iterrows():
        print(f"  {row['country']} {int(row['year'])}  "
              f"IS={row['IS_agg']:.4f}  "
              f"true={row['true_zone']}  pred={row['pred_zone']}  "
              f"override={row['electoral_override']}")

# ── SAVE OUTPUTS ─────────────────────────────────────────────────────────────
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")

# Summary table
summary_rows = []
for zone in all_zones:
    m = zone_metrics[zone]
    summary_rows.append({
        "zone"     : zone,
        "precision": m["precision"],
        "recall"   : m["recall"],
        "f1"       : m["f1"],
        "support"  : m["support"],
    })
summary_rows.append({
    "zone"     : "MACRO",
    "precision": round(np.mean([zone_metrics[z]["precision"] for z in zones_present]), 4),
    "recall"   : round(np.mean([zone_metrics[z]["recall"]    for z in zones_present]), 4),
    "f1"       : round(f1_macro, 4),
    "support"  : n,
})
summary_rows.append({
    "zone"     : "ACCURACY",
    "precision": "",
    "recall"   : "",
    "f1"       : round(overall_accuracy, 4),
    "support"  : n,
})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"Saved: {SUMMARY_CSV}")
