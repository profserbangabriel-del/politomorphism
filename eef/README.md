# EEF — Entropic Equilibrium Function

**Politomorphism Framework** | Serban Gabriel Florin  
ORCID: 0009-0000-2266-3356 | OSF: https://doi.org/10.17605/OSF.IO/HYDNZ  
GitHub: https://github.com/profserbangabriel-del/Politomorphism

---

## Overview

The **Entropic Equilibrium Function (EEF)** measures political systemic instability as Shannon entropy over institutional state distributions. It produces a static score S(t)/S_max per domain, an aggregate entropy score with zone classification, and a sensitivity analysis.

---

## Formula

    S(t) = -sum( p_i(t) * log(p_i(t)) )   [Shannon entropy]
    S_max = log(N)                          [maximum entropy for N states]
    ratio = S(t) / S_max                    [normalized instability score, range 0-1]

    dS/dt = Pi(t) - Phi(t)                 [entropy dynamics]
      Pi(t)  = disorder-generating forces (crises, conflicts, disruptive events)
      Phi(t) = order-restoring forces (reforms, institutional adaptation)

    Equilibrium condition:  dS/dt = 0  =>  Pi(t) = Phi(t)
    Optimal stability:      S* = arg min |dS/dt|

---

## Entropy zones

| S(t)/S_max | Zone     | Political interpretation                                          |
|------------|----------|------------------------------------------------------------------|
| > 80%      | CRITICAL | Structural instability; disorder production exceeds self-regulation |
| 60-80%     | HIGH     | Significant institutional fragmentation; reform capacity under strain |
| 40-60%     | MODERATE | Manageable tensions; institutional stress present but containable |
| < 40%      | LOW      | System near equilibrium; order-sustaining mechanisms functional  |

---

## State space (three domains, three states each)

| Domain    | State 1                  | State 2              | State 3                    |
|-----------|--------------------------|----------------------|----------------------------|
| Justice   | Functional independence  | Political capture    | Paralysis/vacuum           |
| Electoral | Legitimate functioning   | Crisis/contested     | Delegitimized/captured     |
| Coalition | Stable and coherent      | Fragile/conflictual  | Collapse/reconfiguration   |

---

## Usage

Run with a config file:

    python compute_eef.py --config config_eef_romania.json

Output example (Romania 2024):

    ============================================================
      EEF: Romania 2024
    ============================================================
      Domain           S(t)    S_max        %  Zone
      ------------------------------------------------------
      Justice          0.9973   1.0986    90.8%  CRITICAL
      Electoral        0.9086   1.0986    82.7%  CRITICAL
      Coalition        0.8865   1.0986    80.7%  CRITICAL
      ------------------------------------------------------
      AGGREGATE                           84.7%  CRITICAL

      SENSITIVITY ANALYSIS
      Scenario     Justice  Electoral  Coalition  Aggregate  Zone
      -20%          99.6%     85.0%     96.8%      93.8%  CRITICAL
      -10%          97.0%     85.7%     90.6%      91.1%  CRITICAL
      +0%           90.8%     82.7%     80.7%      84.7%  CRITICAL  <-- baseline
      +10%          80.8%     76.0%     66.7%      74.5%  HIGH
      +20%          66.8%     65.5%     47.8%      60.0%  HIGH
    ============================================================

---

## Cross-national results (2024)

| Country | Justice | Electoral | Coalition | Aggregate | Zone     |
|---------|---------|-----------|-----------|-----------|----------|
| Romania | 90.8%   | 82.7%     | 80.7%     | 84.7%     | CRITICAL |
| Hungary | 73.0%   | 94.6%     | 62.6%     | 76.7%     | HIGH     |
| Poland  | 97.1%   | 84.3%     | 88.7%     | 90.1%     | CRITICAL |

Sources: Freedom House Nations in Transit 2024; BTI Transformation Index 2026;
EC Rule of Law Report 2024; Venice Commission 2024; INSCOP Research January 2026;
EU Justice Scoreboard 2025.

---

## Files

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| compute_eef.py            | Main script — computes EEF scores, sensitivity   |
| config_eef_romania.json   | Romania 2024 baseline calibration with sources   |
| config_eef_hungary.json   | Hungary 2024 cross-national validation           |
| config_eef_poland.json    | Poland 2024 cross-national validation            |

---

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

---

## Requirements

Python >= 3.10. No external dependencies.

---

## Citation

Florin, S.G. (2026). Politomorphism and the Measurement of Political Systemic
Instability: The Entropic Equilibrium Function (EEF).
Politomorphism Framework Working Paper.
OSF: https://doi.org/10.17605/OSF.IO/HYDNZ

---

## License

CC BY 4.0 — Open for replication, extension, and empirical validation.
