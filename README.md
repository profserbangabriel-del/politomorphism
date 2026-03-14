# Politomorphism — Social Resonance Model (SRM)

**Serban Gabriel Florin** | Independent Researcher  
ORCID: [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356) | DOI: [10.17605/OSF.IO/HYDNZ](https://doi.org/10.17605/OSF.IO/HYDNZ)  
GitHub: [profserbangabriel-del/politomorphism](https://github.com/profserbangabriel-del/politomorphism)

---

## What is Politomorphism?

Politomorphism is a theoretical framework that treats political symbols as **morphogenetic agents** — entities that transform power structures through the process of symbolic diffusion. Its computational component, the **Social Resonance Model (SRM)**, provides a quantitative measure of this phenomenon.

## SRM Formula

**SRM = V × A × e^(−λD) × N** (λ = 2)

| Variable | Name | Definition |
|----------|------|-----------|
| V | Viral Velocity | Rate of symbol diffusion (log-normalized escalation ratio) |
| A | Affective Weight | Emotional intensity (VADER sentiment analysis) |
| D | Semantic Drift | Meaning fragmentation across contexts (0=stable, 1=fully fragmented) |
| N | Network Coverage | Proportion of days/sources where symbol is present in corpus |

**Score interpretation:**
- SRM ≥ 0.20 → HIGH RESONANCE
- 0.07 ≤ SRM < 0.20 → MEDIUM RESONANCE
- 0.02 ≤ SRM < 0.07 → LOW RESONANCE
- SRM < 0.02 → MINIMAL RESONANCE

---

## Comparative Dataset — 5 Validated Symbols

| Symbol / Context | V | A | D | N | **SRM** | Interpretation |
|-----------------|---|---|---|---|---------|----------------|
| Sunflower Mvt (TW, 2014) | 0.680 | 0.420 | 0.7737 | 0.580 | 0.0352 | Low |
| Călin Georgescu (RO, 2024) | 0.750 | 0.398 | 0.8813 | 0.600 | 0.0307 | Low |
| Marcel Ciolacu (RO, 2025-26) | 0.720 | 0.420 | 0.8412 | 0.650 | 0.0365 | Low |
| Donald Trump (US, 2015-16) | 0.958 | 0.580 | 0.7340 | 0.720 | 0.0922 | **Medium** |
| **Zelensky (UA/EU/US, 2022-26)** | **0.873** | **0.640** | **0.680** | **0.781** | **0.1121** | **Medium** |

> **Consistent pattern:** Semantic Drift (D) is the primary differentiating variable across all five cases. Lower D → higher SRM, regardless of visibility level.

---

## Case Studies

### Case Study 1 — Sunflower Movement (Taiwan, 2014)

| V | A | D | N | **SRM** | Interpretation |
|---|---|---|---|---------|----------------|
| 0.680 | 0.420 | 0.7737 | 0.580 | **0.0352** | LOW RESONANCE |

---

### Case Study 2 — Călin Georgescu (Romania, Oct–Dec 2024)

| V | A | D | N | **SRM** | Interpretation |
|---|---|---|---|---------|----------------|
| 0.750 | 0.398 | 0.8813 | 0.600 | **0.0307** | LOW RESONANCE |

Results: [SRM_raport_final.json](SRM_raport_final.json) | Chart: [SRM_grafic_final.png](SRM_grafic_final.png)

---

### Case Study 3 — Marcel Ciolacu (Romania, Dec 2025 – Mar 2026)
Data: Media Cloud Romania National + State & Local | 539 articles | 50 sources | 91 days

| V | A | D | N | **SRM** | Interpretation |
|---|---|---|---|---------|----------------|
| 0.720 | 0.420 | 0.8412 | 0.650 | **0.0365** | LOW RESONANCE |

**Key finding — The Resonance Paradox:** High visibility (V=0.72, N=0.65), low resonance due to extreme semantic drift (D=0.8412). Six competing semantic frames: opposition critic, former premier blamed for crisis, local administrator, PSD party leader, judicial target, electoral strategist.

**New diagnostic category:** *Post-Executive Symbolic Trap* — the transition from prime minister to opposition generates maximum semantic fragmentation as a structural consequence.

Paper: [SRM_Ciolacu_Validation.docx](SRM_Ciolacu_Validation.docx) | Data: [data_ciolacu/](data_ciolacu/)

---

### Case Study 4 — Donald Trump (USA, Feb 2015 – Nov 2016)
Data: Media Cloud US National + State & Local | 640 daily observations

| V | A | D | N | **SRM** | Interpretation |
|---|---|---|---|---------|----------------|
| 0.958 | 0.580 | 0.7340 | 0.720 | **0.0922** | MEDIUM RESONANCE |

**Key finding:** First confirmed case of Medium Resonance. V=0.958 — 82.3x escalation from pre-campaign baseline. D=0.734 — semantically coherent enough for successful electoral mobilization.

Results: [SRM_trump_result.json](SRM_trump_result.json) | Chart: [SRM_trump_grafic.png](SRM_trump_grafic.png)

---

### Case Study 5 — Volodymyr Zelensky (UA/EU/US, May 2022 – Feb 2026)
Data: Media Cloud US National + Europe | 2,387 daily observations (baseline + analysis period)

| V | A | D | N | **SRM** | Interpretation |
|---|---|---|---|---------|----------------|
| 0.873 | 0.640 | 0.680 | 0.781 | **0.1121** | MEDIUM RESONANCE |

**Key finding:** Highest SRM score in the current dataset. Escalation of **123.6x** in US media from pre-invasion baseline (May 2019 – Feb 2022). Peak event: **February 28 – March 4, 2025** (Zelensky–Trump meeting at the White House). D=0.680 — lowest in dataset, reflecting dominance of the "resistance leader" frame across Western media.

**New diagnostic category:** *Sustained Wartime Medium-Resonance Symbol* — multi-year high visibility, cross-cultural presence, semantic coherence maintained through the dominance of a single primary frame.

Paper: [SRM_Zelensky_Validation.docx](SRM_Zelensky_Validation.docx) | Results: [SRM_zelensky_result.json](SRM_zelensky_result.json) | Data: [data_zelensky/](data_zelensky/)

---

## Diagnostic Categories (SRM Typology)

| Category | Symbol | Key Feature |
|----------|--------|-------------|
| Fragmented Diffusion Symbol | Georgescu | Extreme D (0.8813), ideological fragmentation |
| Post-Executive Symbolic Trap | Ciolacu | High D (0.8412), institutional role transition |
| High-Velocity Campaign Symbol | Trump | Highest V (0.958), event-driven peaks |
| Sustained Wartime Medium-Resonance Symbol | Zelensky | Lowest D (0.680), wartime coherence |

---

## Repository Structure

```
politomorphism/
├── .github/workflows/
│   ├── srm_zelensky_validation.yml
│   └── srm_sunflower_validation.yml
├── data_zelensky/
│   ├── counts_zelensky_first_period.csv
│   ├── counts_zelensky_second_period.csv
│   ├── counts_zelensky_usa.csv
│   └── counts_zelensky_europe.csv
├── data_ciolacu/
├── data_sunflower/
├── SRM_Zelensky_Validation.docx
├── SRM_Ciolacu_Validation.docx
├── SRM_zelensky_result.json
├── SRM_trump_result.json
├── SRM_raport_final.json
└── index.html
```

---

## Reproducibility

All data, code, and results are open-source and publicly available.  
Data source: [mediacloud.org](https://mediacloud.org)  
License: CC BY 4.0
