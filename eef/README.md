# EEF — Entropic Equilibrium Function

Politomorphism Framework | Serban Gabriel Florin
ORCID: 0009-0000-2266-3356 | OSF: https://doi.org/10.17605/OSF.IO/HYDNZ
GitHub: https://github.com/profserbangabriel-del/Politomorphism

## Overview

The Entropic Equilibrium Function (EEF) measures political systemic instability as Shannon entropy over institutional state distributions. It produces a static score H(t) per domain, an aggregate instability score with zone classification, and a sensitivity analysis.

FIIM v2.1 (Fuzzy Institutional Instability Model) extends EEF by replacing hard threshold mapping with continuous fuzzy membership functions and combining normalized entropy with an expected institutional cost index.

## Notation

| Symbol | Definition |
|---|---|
| S(t) | Raw Shannon entropy: S(t) = -sum( p_i(t) * ln(p_i(t)) ) [nats] |
| S_max | Maximum entropy: S_max = ln(N) = ln(3) ≈ 1.0986 nats |
| H(t) | Normalized entropy: H(t) = S(t) / S_max, range 0-1 |
| V(t) | Expected institutional cost: V(t) = 0.0*p0 + 0.5*p1 + 1.0*p2 |
| IS(t) | Static instability score: IS(t) = alpha*H(t) + (1-alpha)*V(t) |
| Delta_IS(t) | Discrete dynamics: Delta_IS(t) = IS(t) - IS(t-1) |
| Pi(t) | Entropy-increasing component: Pi(t) = max(0, Delta_IS(t)) |
| Phi(t) | Entropy-decreasing component: Phi(t) = max(0, -Delta_IS(t)) |
| IS_comp(t) | Composite score: IS_comp(t) = IS(t) + beta*Delta_IS(t) |

## Formula

S(t)     = -sum( p_i(t) * ln(p_i(t)) )     raw Shannon entropy [nats, base e]
S_max    = ln(3) ≈ 1.0986                   maximum entropy for N=3 states
H(t)     = S(t) / S_max                     normalized entropy, base-invariant
V(t)     = 0.0*p0 + 0.5*p1 + 1.0*p2        expected institutional cost
IS(t)    = 0.5 * H(t) + 0.5 * V(t)         static instability score
Delta_IS = IS(t) - IS(t-1)                  discrete dynamics (annual)
Pi(t)    = max(0, Delta_IS)                 disorder-increasing component
Phi(t)   = max(0, -Delta_IS)                order-restoring component
IS_comp  = IS(t) + 0.2 * Delta_IS(t)        composite forward-looking score

Note: Shannon entropy is computed on discrete annual distributions.
Continuous-time notation dS/dt is replaced by discrete Delta_IS(t).
Normalization H(t) = S(t)/S_max cancels the logarithmic base.

## V(t) — Expected Institutional Cost

V(t) = E[c(X)] where c is an ordinal cost function defined on the state space:

| State | Label | Cost c(Si) | Interpretation |
|---|---|---|---|
| S0 | Stable | 0.0 | Zero institutional cost — functional equilibrium |
| S1 | Strained | 0.5 | Partial dysfunction — reform capacity under pressure |
| S2 | Critical | 1.0 | Full institutional failure — self-regulation exhausted |

## Instability zones

| IS(t) | Zone | Interpretation |
|---|---|---|
| > 0.70 | CRITICAL | Structural instability — disorder exceeds self-regulation |
| 0.55-0.70 | HIGH | Significant fragmentation — reform capacity under strain |
| 0.40-0.55 | MODERATE | Manageable tensions — stress containable |
| < 0.40 | LOW | System near equilibrium |

## State space (three domains, three states each)

| Domain | S0 — Stable | S1 — Strained | S2 — Critical |
|---|---|---|---|
| Justice | Functional independence | Political capture | Paralysis/vacuum |
| Electoral | Legitimate functioning | Crisis/contested | Delegitimized/captured |
| Coalition | Stable and coherent | Fragile/conflictual | Collapse/reconfiguration |

## Cross-national results (2024)

| Country | Justice | Electoral | Coalition | Aggregate | Zone |
|---|---|---|---|---|---|
| Romania | 90.8% | 82.7% | 80.7% | 84.7% | CRITICAL |
| Hungary | 73.0% | 94.6% | 62.6% | 76.7% | HIGH |
| Poland | 97.1% | 84.3% | 88.7% | 90.1% | CRITICAL |

Sources: Freedom House NIT 2024; BTI 2026; EC Rule of Law 2024; Venice Commission 2024; INSCOP January 2026; EU Justice Scoreboard 2025.

## Files

| File | Description |
|---|---|
| compute_eef.py | Main script — computes EEF scores, sensitivity |
| config_eef_romania.json | Romania 2024 baseline calibration with sources |
| config_eef_hungary.json | Hungary 2024 cross-national validation |
| config_eef_poland.json | Poland 2024 cross-national validation |
| eef_longitudinal.py | Longitudinal validation 2005-2024 (FH NIT + BTI) |
| eef_interrater.py | Inter-rater reliability — Krippendorff alpha + Cohen kappa |
| eef_fiim.py | FIIM v2.1 — fuzzy membership, H(t), V(t), IS_comp |
| eef_comparison_table.py | EEF vs FIIM comparison — hard vs fuzzy thresholds |
| eef_bootstrap.py | Bootstrap 95% CI for FIIM IS scores (n=1000) |

## Usage

Run EEF original:
python compute_eef.py --config config_eef_romania.json

Run FIIM v2.1:
python eef_fiim.py

Run longitudinal validation:
python eef_longitudinal.py

Run inter-rater reliability:
python eef_interrater.py

Run bootstrap CI:
python eef_bootstrap.py

## Adding a new country

Create a JSON config file with this structure:

{
  "country": "CountryName",
  "year": "2024",
  "domains": {
    "Justice":   [p_functional, p_capture, p_paralysis],
    "Electoral": [p_legitimate, p_crisis, p_delegitimized],
    "Coalition": [p_stable, p_fragile, p_collapse]
  },
  "sources": {
    "Justice":   ["Source 1", "Source 2"],
    "Electoral": ["Source 1", "Source 2"],
    "Coalition": ["Source 1", "Source 2"]
  }
}

Then run:
python compute_eef.py --config config_eef_yourcountry.json

## Requirements

Python >= 3.10. No external dependencies.

## Citation

Florin, S.G. (2026). Politomorphism and the Measurement of Political Systemic Instability: The Entropic Equilibrium Function (EEF). Politomorphism Framework Working Paper. OSF: https://doi.org/10.17605/OSF.IO/HYDNZ

## License

CC BY 4.0 — Open for replication, extension, and empirical validation.
