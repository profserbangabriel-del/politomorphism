"""
EEF Bootstrap Confidence Intervals — Politomorphism Engine
===========================================================
Computes 95% bootstrap confidence intervals for IS_agg (FIIM v2.0)
by resampling the probability vectors p_d(t) with perturbation.

Method:
  For each country-year observation:
  1. Generate n_bootstrap samples by adding random noise to raw scores
     drawn from Uniform(-delta, +delta) where delta = 0.25 (one quarter-point)
  2. Recompute p_d(t) via fuzzy membership for each sample
  3. Compute IS_agg for each sample
  4. Report mean, std, 2.5th and 97.5th percentiles

Author : Prof. Serban Gabriel Florin | ORCID: 0009-0000-2266-3356
Project: Politomorphism Engine | OSF: 10.17605/OSF.IO/HYDNZ
"""

import math
import random
import csv

random.seed(42)

# ── FUZZY FUNCTIONS ───────────────────────────────────────────────────────────

def smf(x, a, b):
    if x <= a: return 0.0
    if x >= b: return 1.0
    if x <= (a + b) / 2: return 2 * ((x - a) / (b - a)) ** 2
    return 1 - 2 * ((x - b) / (b - a)) ** 2

def zmf(x, a, b): return 1.0 - smf(x, a, b)

def trimf_clipped(x, center, hw):
    return max(0.0, 1.0 - abs(x - center) / hw)

def normalize(mus):
    t = sum(mus) + 1e-10
    return [m / t for m in mus]

def fuzzy_fh_judicial(s):
    return normalize([smf(s,3.0,6.0), trimf_clipped(s,4.0,3.5), zmf(s,1.0,5.0)])

def fuzzy_fh_electoral(s):
    return normalize([smf(s,3.0,6.0), trimf_clipped(s,3.75,3.25), zmf(s,1.0,5.0)])

def fuzzy_bti(s):
    return normalize([smf(s,4.5,9.0), trimf_clipped(s,6.0,5.0), zmf(s,1.0,7.5)])

def H(p): return (-sum(x*math.log(x) for x in p if x>0)) / math.log(3)
def V(p): return 0.5*p[1] + 1.0*p[2]
def IS(p, alpha=0.5): return alpha*H(p) + (1-alpha)*V(p)

def is_zone(s):
    if s > 0.70:   return "CRITICAL"
    elif s > 0.55: return "HIGH"
    elif s > 0.40: return "MODERATE"
    else:          return "LOW"

# ── OVERRIDES ─────────────────────────────────────────────────────────────────

OVERRIDES = {
    ("Hungary",  2011): [0.05, 0.45, 0.50],
    ("Hungary",  2014): [0.05, 0.40, 0.55],
    ("Poland",   2015): [0.20, 0.55, 0.25],
    ("Poland",   2019): [0.08, 0.48, 0.44],
    ("Romania",  2024): [0.08, 0.52, 0.40],
}

# ── DATA ──────────────────────────────────────────────────────────────────────

FH_JUDICIAL = {
    "Romania": {
        2005:3.50,2006:3.50,2007:3.75,2008:3.75,2009:4.00,
        2010:4.00,2011:3.75,2012:3.75,2013:4.00,2014:4.00,
        2015:4.25,2016:4.25,2017:4.00,2018:3.75,2019:3.75,
        2020:4.00,2021:4.00,2022:4.00,2023:4.25,2024:4.50,
    },
    "Hungary": {
        2005:4.25,2006:4.25,2007:4.25,2008:4.25,2009:4.25,
        2010:4.00,2011:3.50,2012:3.00,2013:2.75,2014:2.50,
        2015:2.25,2016:2.00,2017:2.00,2018:2.00,2019:2.00,
        2020:2.00,2021:1.75,2022:1.75,2023:1.75,2024:1.75,
    },
    "Poland": {
        2005:4.25,2006:4.25,2007:4.50,2008:4.50,2009:4.50,
        2010:4.50,2011:4.50,2012:4.75,2013:4.75,2014:4.75,
        2015:4.75,2016:4.25,2017:3.75,2018:3.50,2019:3.25,
        2020:3.00,2021:3.00,2022:3.00,2023:3.25,2024:3.50,
    },
}

FH_ELECTORAL = {
    "Romania": {
        2005:3.25,2006:3.25,2007:3.50,2008:3.50,2009:3.50,
        2010:3.50,2011:3.50,2012:3.50,2013:3.75,2014:3.75,
        2015:3.75,2016:3.75,2017:3.75,2018:3.75,2019:3.75,
        2020:3.75,2021:3.75,2022:3.75,2023:3.75,2024:3.25,
    },
    "Hungary": {
        2005:4.00,2006:4.00,2007:4.00,2008:4.00,2009:3.75,
        2010:3.75,2011:3.50,2012:3.25,2013:3.00,2014:2.75,
        2015:2.75,2016:2.75,2017:2.50,2018:2.50,2019:2.50,
        2020:2.50,2021:2.25,2022:2.25,2023:2.25,2024:2.00,
    },
    "Poland": {
        2005:4.50,2006:4.50,2007:4.75,2008:4.75,2009:4.75,
        2010:4.75,2011:4.75,2012:4.75,2013:4.75,2014:4.75,
        2015:4.75,2016:4.50,2017:4.25,2018:4.00,2019:4.00,
        2020:3.75,2021:3.75,2022:3.75,2023:3.75,2024:4.00,
    },
}

BTI_BIENNIAL = {
    "Romania": {2006:7.5,2008:7.5,2010:7.5,2012:7.0,2014:7.5,2016:7.5,2018:7.5,2020:7.0,2022:7.0,2024:7.0},
    "Hungary": {2006:8.5,2008:8.5,2010:8.0,2012:7.0,2014:6.0,2016:5.5,2018:5.0,2020:4.5,2022:4.5,2024:4.5},
    "Poland":  {2006:9.0,2008:9.0,2010:9.0,2012:9.0,2014:9.0,2016:8.5,2018:7.5,2020:7.0,2022:7.0,2024:7.5},
}

YEARS     = list(range(2005, 2025))
COUNTRIES = ["Romania", "Hungary", "Poland"]
ALPHA     = 0.5
N_BOOT    = 1000
DELTA     = 0.25


def interpolate_bti(country, year):
    data = BTI_BIENNIAL[country]
    if year in data: return data[year]
    ys = sorted(data.keys())
    for i in range(len(ys)-1):
        y0, y1 = ys[i], ys[i+1]
        if y0 < year < y1:
            return data[y0] + (data[y1]-data[y0])*(year-y0)/(y1-y0)
    return data[ys[0]] if year < ys[0] else data[ys[-1]]


def percentile(data, pct):
    data_sorted = sorted(data)
    idx = (len(data_sorted) - 1) * pct / 100
    lo, hi = int(idx), min(int(idx)+1, len(data_sorted)-1)
    return data_sorted[lo] + (data_sorted[hi]-data_sorted[lo])*(idx-lo)


def bootstrap_IS(country, year, jud, elec, bti):
    """Generate N_BOOT bootstrap samples and return IS distribution."""
    samples = []
    is_override = (country, year) in OVERRIDES

    for _ in range(N_BOOT):
        # Perturb raw scores with uniform noise
        jud_s  = max(1.0, min(7.0,  jud  + random.uniform(-DELTA, DELTA)))
        elec_s = max(1.0, min(7.0,  elec + random.uniform(-DELTA, DELTA)))
        bti_s  = max(1.0, min(10.0, bti  + random.uniform(-DELTA, DELTA)))

        p_j = fuzzy_fh_judicial(jud_s)
        # Overrides are not perturbed — they are deterministic events
        p_e = OVERRIDES.get((country, year), fuzzy_fh_electoral(elec_s))
        p_c = fuzzy_bti(bti_s)

        IS_agg = (IS(p_j, ALPHA) + IS(p_e, ALPHA) + IS(p_c, ALPHA)) / 3
        samples.append(IS_agg)

    return samples


# ── MAIN ──────────────────────────────────────────────────────────────────────

def compute_bootstrap():
    results = []

    for country in COUNTRIES:
        for year in YEARS:
            jud  = FH_JUDICIAL[country].get(year)
            elec = FH_ELECTORAL[country].get(year)
            bti  = interpolate_bti(country, year)
            if jud is None or elec is None:
                continue

            # Point estimate (no perturbation)
            p_j = fuzzy_fh_judicial(jud)
            p_e = OVERRIDES.get((country, year), fuzzy_fh_electoral(elec))
            p_c = fuzzy_bti(bti)
            IS_point = (IS(p_j,ALPHA) + IS(p_e,ALPHA) + IS(p_c,ALPHA)) / 3

            # Bootstrap distribution
            samples = bootstrap_IS(country, year, jud, elec, bti)

            mean_IS = sum(samples) / len(samples)
            std_IS  = (sum((s-mean_IS)**2 for s in samples)/len(samples))**0.5
            ci_lo   = percentile(samples, 2.5)
            ci_hi   = percentile(samples, 97.5)

            # Zone stability: fraction of samples in same zone as point estimate
            point_zone = is_zone(IS_point)
            zone_stable = sum(1 for s in samples if is_zone(s)==point_zone) / len(samples)

            results.append({
                "country":     country,
                "year":        year,
                "IS_point":    round(IS_point*100, 2),
                "IS_mean":     round(mean_IS*100, 2),
                "IS_std":      round(std_IS*100, 2),
                "CI_lo":       round(ci_lo*100, 2),
                "CI_hi":       round(ci_hi*100, 2),
                "CI_width":    round((ci_hi-ci_lo)*100, 2),
                "zone":        point_zone,
                "zone_stable": round(zone_stable*100, 1),
                "override":    (country, year) in OVERRIDES,
            })

    return results


def print_results(results):
    for country in COUNTRIES:
        rows = [r for r in results if r["country"]==country]
        print(f"\n{'='*80}")
        print(f"  BOOTSTRAP CI — {country.upper()}  |  n={N_BOOT}, delta=±{DELTA}")
        print(f"{'='*80}")
        print(f"  {'Year':>4}  {'IS%':>6}  {'Mean':>6}  {'Std':>5}  "
              f"{'CI 2.5%':>7}  {'CI 97.5%':>8}  {'Width':>6}  "
              f"{'Zone':>10}  {'Stable%':>8}")
        print(f"  {'─'*78}")
        for r in rows:
            ovr = "*" if r["override"] else " "
            print(f"  {r['year']:>4}  {r['IS_point']:>5.1f}%  "
                  f"{r['IS_mean']:>5.1f}%  {r['IS_std']:>4.2f}  "
                  f"{r['CI_lo']:>6.1f}%  {r['CI_hi']:>7.1f}%  "
                  f"{r['CI_width']:>5.1f}%  "
                  f"{r['zone']:>10}{ovr} {r['zone_stable']:>7.1f}%")


def print_summary(results):
    print(f"\n{'='*65}")
    print(f"  SUMMARY — Zone Stability & CI Width")
    print(f"{'='*65}")
    print(f"  {'Country':<10} {'Year':>4}  {'IS%':>6}  "
          f"[{'95% CI':^15}]  {'Stable%':>8}  Zone")
    print(f"  {'─'*63}")

    # Print only key years
    key_years = [2005, 2010, 2015, 2018, 2019, 2021, 2024]
    for r in results:
        if r["year"] in key_years:
            ovr = "*" if r["override"] else " "
            print(f"  {r['country']:<10} {r['year']:>4}  "
                  f"{r['IS_point']:>5.1f}%  "
                  f"[{r['CI_lo']:>5.1f}% – {r['CI_hi']:>5.1f}%]  "
                  f"{r['zone_stable']:>7.1f}%  {r['zone']}{ovr}")


def export_csv(results, path="EEF_Bootstrap_CI.csv"):
    if not results: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  CSV exported: {path}")


if __name__ == "__main__":
    print("\n  Politomorphism Engine — Bootstrap Confidence Intervals")
    print(f"  Method : Uniform perturbation ±{DELTA} on raw scores")
    print(f"  Samples: n={N_BOOT} per country-year observation")
    print(f"  Alpha  : {ALPHA} (entropy/severity weight)")
    print(f"  Countries: {', '.join(COUNTRIES)}")
    print(f"  Period   : {YEARS[0]}–{YEARS[-1]}")

    results = compute_bootstrap()
    print_results(results)
    print_summary(results)
    export_csv(results)
    print("\n  Done.\n")
