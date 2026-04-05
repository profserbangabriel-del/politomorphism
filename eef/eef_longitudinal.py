"""
EEF Longitudinal Validation — Politomorphism Engine
=====================================================
Computes EEF scores (R_d, R_EEF) and entropic dynamics (Pi, Phi, dS/dt)
for Romania, Hungary, and Poland across 2006-2024.

Data sources:
  FH NIT  — Freedom House Nations in Transit, Democracy Score (1-7 scale)
  BTI     — Bertelsmann Transformation Index, Political Participation (1-10)

Author : Prof. Serban Gabriel Florin | ORCID: 0009-0000-2266-3356
Project: Politomorphism Engine | OSF: 10.17605/OSF.IO/HYDNZ
"""

import math
import csv

# ── RAW DATA ──────────────────────────────────────────────────────────────────

FH_JUDICIAL = {
    "Romania": {
        2005:3.50, 2006:3.50, 2007:3.75, 2008:3.75, 2009:4.00,
        2010:4.00, 2011:3.75, 2012:3.75, 2013:4.00, 2014:4.00,
        2015:4.25, 2016:4.25, 2017:4.00, 2018:3.75, 2019:3.75,
        2020:4.00, 2021:4.00, 2022:4.00, 2023:4.25, 2024:4.50,
    },
    "Hungary": {
        2005:4.25, 2006:4.25, 2007:4.25, 2008:4.25, 2009:4.25,
        2010:4.00, 2011:3.50, 2012:3.00, 2013:2.75, 2014:2.50,
        2015:2.25, 2016:2.00, 2017:2.00, 2018:2.00, 2019:2.00,
        2020:2.00, 2021:1.75, 2022:1.75, 2023:1.75, 2024:1.75,
    },
    "Poland": {
        2005:4.25, 2006:4.25, 2007:4.50, 2008:4.50, 2009:4.50,
        2010:4.50, 2011:4.50, 2012:4.75, 2013:4.75, 2014:4.75,
        2015:4.75, 2016:4.25, 2017:3.75, 2018:3.50, 2019:3.25,
        2020:3.00, 2021:3.00, 2022:3.00, 2023:3.25, 2024:3.50,
    },
}

FH_ELECTORAL = {
    "Romania": {
        2005:3.25, 2006:3.25, 2007:3.50, 2008:3.50, 2009:3.50,
        2010:3.50, 2011:3.50, 2012:3.50, 2013:3.75, 2014:3.75,
        2015:3.75, 2016:3.75, 2017:3.75, 2018:3.75, 2019:3.75,
        2020:3.75, 2021:3.75, 2022:3.75, 2023:3.75, 2024:3.25,
    },
    "Hungary": {
        2005:4.00, 2006:4.00, 2007:4.00, 2008:4.00, 2009:3.75,
        2010:3.75, 2011:3.50, 2012:3.25, 2013:3.00, 2014:2.75,
        2015:2.75, 2016:2.75, 2017:2.50, 2018:2.50, 2019:2.50,
        2020:2.50, 2021:2.25, 2022:2.25, 2023:2.25, 2024:2.00,
    },
    "Poland": {
        2005:4.50, 2006:4.50, 2007:4.75, 2008:4.75, 2009:4.75,
        2010:4.75, 2011:4.75, 2012:4.75, 2013:4.75, 2014:4.75,
        2015:4.75, 2016:4.50, 2017:4.25, 2018:4.00, 2019:4.00,
        2020:3.75, 2021:3.75, 2022:3.75, 2023:3.75, 2024:4.00,
    },
}

BTI_PARTICIPATION_BIENNIAL = {
    "Romania": {
        2006:7.5, 2008:7.5, 2010:7.5, 2012:7.0, 2014:7.5,
        2016:7.5, 2018:7.5, 2020:7.0, 2022:7.0, 2024:7.0,
    },
    "Hungary": {
        2006:8.5, 2008:8.5, 2010:8.0, 2012:7.0, 2014:6.0,
        2016:5.5, 2018:5.0, 2020:4.5, 2022:4.5, 2024:4.5,
    },
    "Poland": {
        2006:9.0, 2008:9.0, 2010:9.0, 2012:9.0, 2014:9.0,
        2016:8.5, 2018:7.5, 2020:7.0, 2022:7.0, 2024:7.5,
    },
}

YEARS = list(range(2005, 2025))
COUNTRIES = ["Romania", "Hungary", "Poland"]

ELECTORAL_OVERRIDES = {
    ("Hungary",  2011): [0.05, 0.45, 0.50],
    ("Hungary",  2014): [0.05, 0.40, 0.55],
    ("Poland",   2015): [0.20, 0.55, 0.25],
    ("Poland",   2019): [0.08, 0.48, 0.44],
    ("Romania",  2024): [0.08, 0.52, 0.40],
}

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def interpolate_bti(country, year):
    data = BTI_PARTICIPATION_BIENNIAL[country]
    if year in data:
        return data[year]
    years_sorted = sorted(data.keys())
    for i in range(len(years_sorted) - 1):
        y0, y1 = years_sorted[i], years_sorted[i + 1]
        if y0 < year < y1:
            return data[y0] + (data[y1] - data[y0]) * (year - y0) / (y1 - y0)
    if year < years_sorted[0]:
        return data[years_sorted[0]]
    return data[years_sorted[-1]]

def fh_judicial_to_probs(score):
    if score >= 5.5:  return [0.70, 0.25, 0.05]
    elif score >= 5.0: return [0.55, 0.35, 0.10]
    elif score >= 4.5: return [0.40, 0.45, 0.15]
    elif score >= 4.0: return [0.25, 0.55, 0.20]
    elif score >= 3.5: return [0.15, 0.55, 0.30]
    elif score >= 3.0: return [0.10, 0.50, 0.40]
    elif score >= 2.5: return [0.05, 0.40, 0.55]
    elif score >= 2.0: return [0.03, 0.30, 0.67]
    else:              return [0.02, 0.18, 0.80]

def fh_electoral_to_probs(score):
    if score >= 5.5:  return [0.70, 0.25, 0.05]
    elif score >= 5.0: return [0.55, 0.35, 0.10]
    elif score >= 4.5: return [0.40, 0.45, 0.15]
    elif score >= 4.0: return [0.25, 0.52, 0.23]
    elif score >= 3.5: return [0.18, 0.55, 0.27]
    elif score >= 3.0: return [0.12, 0.52, 0.36]
    elif score >= 2.5: return [0.08, 0.48, 0.44]
    elif score >= 2.0: return [0.05, 0.38, 0.57]
    else:              return [0.02, 0.23, 0.75]

def bti_participation_to_probs(score_10):
    if score_10 >= 9.0:  return [0.70, 0.25, 0.05]
    elif score_10 >= 8.0: return [0.55, 0.35, 0.10]
    elif score_10 >= 7.5: return [0.40, 0.48, 0.12]
    elif score_10 >= 7.0: return [0.25, 0.57, 0.18]
    elif score_10 >= 6.5: return [0.18, 0.57, 0.25]
    elif score_10 >= 6.0: return [0.12, 0.55, 0.33]
    elif score_10 >= 5.5: return [0.08, 0.52, 0.40]
    elif score_10 >= 5.0: return [0.06, 0.44, 0.50]
    elif score_10 >= 4.5: return [0.04, 0.35, 0.61]
    else:                 return [0.02, 0.23, 0.75]

def shannon_entropy(probs):
    total = sum(probs)
    p = [x / total for x in probs]
    return -sum(x * math.log(x) for x in p if x > 0)

def eef_score(probs):
    S = shannon_entropy(probs)
    S_max = math.log(len(probs))
    R = S / S_max
    zone = ("CRITICAL" if R > 0.80 else
            "HIGH"     if R > 0.60 else
            "MODERATE" if R > 0.40 else "LOW")
    return {"S": round(S,6), "S_max": round(S_max,6),
            "R": round(R,4), "pct": round(R*100,2), "zone": zone}

# ── MAIN COMPUTATION ──────────────────────────────────────────────────────────

def compute_all():
    results = {}
    for country in COUNTRIES:
        results[country] = {}
        prev_R_EEF = None
        for year in YEARS:
            jud  = FH_JUDICIAL[country].get(year)
            elec = FH_ELECTORAL[country].get(year)
            bti  = interpolate_bti(country, year)
            if jud is None or elec is None:
                continue
            p_justice   = fh_judicial_to_probs(jud)
            p_electoral = ELECTORAL_OVERRIDES.get((country, year),
                          fh_electoral_to_probs(elec))
            p_coalition = bti_participation_to_probs(bti)
            sc_j = eef_score(p_justice)
            sc_e = eef_score(p_electoral)
            sc_c = eef_score(p_coalition)
            R_EEF = (sc_j["R"] + sc_e["R"] + sc_c["R"]) / 3
            zone  = ("CRITICAL" if R_EEF > 0.80 else
                     "HIGH"     if R_EEF > 0.60 else
                     "MODERATE" if R_EEF > 0.40 else "LOW")
            if prev_R_EEF is not None:
                Pi_agg  = max(0.0, R_EEF - prev_R_EEF)
                Phi_agg = 1.0 - R_EEF
                dS_dt   = Pi_agg - Phi_agg
                delta_R = R_EEF - prev_R_EEF
                trend   = ("DETERIORATING" if delta_R >  0.005 else
                           "CONSOLIDATING" if delta_R < -0.005 else
                           "STABLE")
            else:
                Pi_agg = Phi_agg = dS_dt = None
                trend = "N/A (baseline)"
            results[country][year] = {
                "year": year, "country": country,
                "FH_Judicial": jud, "FH_Electoral": elec,
                "BTI_Participation": round(bti, 2),
                "p_Justice": p_justice, "p_Electoral": p_electoral,
                "p_Coalition": p_coalition,
                "R_Justice": sc_j["R"], "pct_Justice": sc_j["pct"],
                "zone_Justice": sc_j["zone"],
                "R_Electoral": sc_e["R"], "pct_Electoral": sc_e["pct"],
                "zone_Electoral": sc_e["zone"],
                "R_Coalition": sc_c["R"], "pct_Coalition": sc_c["pct"],
                "zone_Coalition": sc_c["zone"],
                "R_EEF": round(R_EEF,4), "pct_EEF": round(R_EEF*100,2),
                "zone_EEF": zone,
                "Pi":    round(Pi_agg,4)  if Pi_agg  is not None else None,
                "Phi":   round(Phi_agg,4) if Phi_agg is not None else None,
                "dS_dt": round(dS_dt,4)   if dS_dt   is not None else None,
                "trend": trend,
                "electoral_override": (country, year) in ELECTORAL_OVERRIDES,
            }
            prev_R_EEF = R_EEF
    return results

def export_csv(data, path="EEF_Longitudinal_All.csv"):
    fields = ["country","year","FH_Judicial","FH_Electoral","BTI_Participation",
              "R_Justice","pct_Justice","zone_Justice",
              "R_Electoral","pct_Electoral","zone_Electoral",
              "R_Coalition","pct_Coalition","zone_Coalition",
              "R_EEF","pct_EEF","zone_EEF","Pi","Phi","dS_dt","trend",
              "electoral_override"]
    rows = []
    for country in COUNTRIES:
        for year in YEARS:
            if year in data[country]:
                rows.append(data[country][year])
    rows.sort(key=lambda r: (r["country"], r["year"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"CSV exported: {path}")

if __name__ == "__main__":
    data = compute_all()
    for country in COUNTRIES:
        rows = sorted(data[country].values(), key=lambda r: r["year"])
        print(f"\n{'='*80}\n  {country}\n{'='*80}")
        print(f"  {'Year':>4}  {'FH-J':>5}  {'FH-E':>5}  {'BTI':>5}  "
              f"{'Justice%':>9}  {'Electrl%':>9}  {'Coaltn%':>9}  "
              f"{'R_EEF%':>7}  {'Zone':>10}  Trend")
        for r in rows:
            print(f"  {r['year']:>4}  {r['FH_Judicial']:>5.2f}  "
                  f"{r['FH_Electoral']:>5.2f}  {r['BTI_Participation']:>5.2f}  "
                  f"{r['pct_Justice']:>8.1f}%  {r['pct_Electoral']:>8.1f}%  "
                  f"{r['pct_Coalition']:>8.1f}%  "
                  f"{r['pct_EEF']:>6.1f}%  {r['zone_EEF']:>10}  {r['trend']}")
    export_csv(data)
    print("\nDone.")
