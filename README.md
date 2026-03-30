# Politomorphism — Social Resonance Model (SRM)

**Serban Gabriel Florin** | I:Researcher  
ORCID: [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356) | DOI: [10.17605/OSF.IO/HYDNZ](https://doi.org/10.17605/OSF.IO/HYDNZ)  
GitHub: [profserbangabriel-del/Politomorphism](https://github.com/profserbangabriel-del/Politomorphism)  
License: CC BY 4.0

---

## PE/ICI Correlation Map — 18 Validated Symbol-Periods

[![SRM PE/ICI Correlation Map](docs/srm_pe_ici_map.svg)](https://profserbangabriel-del.github.io/Politomorphism/srm_pe_ici_map_interactive.html)

> **Interactive version:** [srm_pe_ici_map_interactive.html](https://profserbangabriel-del.github.io/Politomorphism/srm_pe_ici_map_interactive.html)  
> **Key finding:** ICI range (0.311) is **4.0× larger** than PE range (0.079) across 18 symbol-periods. Modi 2014 (ICI=0.915) breaks the Western ICI ceiling through CCFI. The CCFI lifecycle (Modi 2014→2019→2024) documents asymptotic stabilization at ~0.80 — irreducible cross-cultural incompatibility. Iran-Israel introduces the first Geopolitical Event Symbol with maximum PE (0.630) through Multi-Domain Geopolitical Activation.

---

## What is Politomorphism?

Politomorphism is a theoretical framework that treats political symbols as **morphogenetic agents** — entities that transform power structures through the process of symbolic diffusion. Its computational component, the **Social Resonance Model (SRM)**, provides a quantitative measure of how effectively a political symbol mobilizes public space.

---

## The SRM Formula

**SRM = V × A × e^(−λD) × N**

| Variable | Name | What it measures | Range |
|----------|------|-----------------|-------|
| V | Viral Velocity | Log-normalized escalation ratio between peak media presence and pre-event baseline | 0–1 |
| A | Affective Weight | Emotional intensity of coverage — computed via VADER (English) or DistilBERT (Romanian) | 0–1 |
| D | Semantic Drift | Fragmentation of meaning. D = α·PE + (1−α)·ICI, α_optimal = **0.338** | 0–1 |
| N | Network Coverage | Proportion of days the symbol appears in corpus | 0–1 |
| **λ** | **Decay Constant** | Controls semantic attenuation speed. Empirically derived from Google Trends. | 2–105 |

---

## How to Read the SRM Score

```
0.00 ────────────── 0.07 ──────────────── 0.20 ──────── 1.00
 LOW RESONANCE          MEDIUM RESONANCE    HIGH RESONANCE
```

> **Empirical finding (18 symbol-periods):** HIGH RESONANCE (>0.20) remains empirically vacant in Western media systems. For Non-Western CCFI symbols, D suppresses SRM regardless of V and A — a phase-specific correction factor is required.

---

## λ Calibration — Step 0 (Mandatory)

> **Key finding:** λ is a typological variable ranging from **λ=2.31** (Orbán) to **λ=104.66** (Charlie Hebdo). All 18 symbols now have empirically calibrated λ values.

### How λ is determined

```
avg / peak = (1 − e^(−λT)) / (λT)
```

Solve for λ using Brent's method (`scipy.optimize.brentq`). **Script:** [`scripts/get_trends.py`](scripts/get_trends.py)

### Complete λ Dataset — 18 Symbol-Periods

| Symbol | λ | e^−λD | Category |
|--------|---|-------|----------|
| Orbán (HU, 2022) | 2.31 | 0.248 | Institutionally Durable |
| **Modi 2014 (IN)** | **2.37** | **0.171** | **Institutionally Durable** |
| Putin (RU, 2022) | 4.90 | 0.032 | Institutionally Durable |
| Zelensky (UA, 2022) | 5.11 | 0.040 | Institutionally Durable |
| **Modi 2019 (IN)** | **6.33** | **0.012** | **Campaign / Ascension** |
| Ciolacu (RO, 2026) | 6.57 | 0.011 | Campaign / Ascension |
| Trump (US, 2016) | 7.01 | 0.008 | Campaign / Ascension |
| **Netanyahu (IL, 2023)** | **7.02** | **0.010** | **Campaign / Ascension** |
| **Modi 2024 (IN)** | **9.11** | **0.002** | **Electorally Volatile** |
| Simion (RO, 2024) | 12.41 | 0.0004 | Electorally Volatile |
| Macron (FR, 2017) | 12.53 | 0.0003 | Electorally Volatile |
| Chávez (VE, 2013) | 16.67 | ≈0 | Electorally Volatile |
| **Iran-Israel (2024)** | **17.81** | **≈0** | **Electorally Volatile** |
| Mandela (ZA, 2013) | 19.66 | ≈0 | Electorally Volatile |
| Georgescu (RO, 2024) | 65.33 | ≈0 | Flash Viral |
| Charlie Hebdo (FR, 2015) | 104.66 | ≈0 | Extreme Flash Viral |

> **Modi λ lifecycle finding:** λ increases across mandates — 2.37 (2014) → 6.33 (2019) → 9.11 (2024) — inversely correlated with CCFI decay. As ICI stabilizes, temporal concentration of interest increases.

> **Recommended default λ = 7.** Flash viral rule: if λ > 30, retain λ=2 and flag as flash event.

---

## D Operationalization

### D = α · PE + (1−α) · ICI

| Component | Name | Measures | Method |
|-----------|------|----------|--------|
| PE | Polysemy Entropy | Topical breadth | Mean Jensen-Shannon Divergence on LDA (K=10, seed=42) |
| ICI | Intra-contextual Incoherence | Framing divergence | 1 − mean pairwise cosine similarity (`paraphrase-multilingual-MiniLM-L12-v2`) |

### Alpha Calibration (14 real-value entries)

| Parameter | Value |
|-----------|-------|
| α_optimal | **0.338** (ICI weight = 0.662) |
| Pearson r (D_new vs D_legacy) | 0.701 |
| Mean D_legacy upward bias | −16.2% |
| ICI range / PE range | **4.0×** |

> **Note:** α = 0.338 calibrated on Western symbols. Phase-specific α recalibration pending for CCFI symbols and Geopolitical Event symbols.

---

## Complete PE/ICI Dataset — 18 Validated Symbol-Periods

| Rank | Symbol | Country | Period | PE | ICI | D_new | ICI−PE | λ | Typology |
|------|--------|---------|--------|----|-----|-------|--------|---|----------|
| 1 | **Modi 2014** | IN | 2013–14 | 0.576 | **0.915** | 0.746 | +0.339 | 2.37 | CCFI acute |
| 2 | Trump | US | 2015–16 | 0.542 | 0.835 | 0.689 | +0.293 | 7.01 | Western ceiling |
| 3 | Mandela | ZA | 2013 | 0.564 | 0.836 | 0.700 | +0.272 | 19.66 | Western ceiling |
| 4 | Putin | RU | 2022 | 0.566 | 0.834 | 0.700 | +0.267 | 4.90 | Western ceiling |
| 5 | **Modi 2024** | IN | 2024 | 0.625 | 0.802 | 0.713 | +0.177 | 9.11 | CCFI asymptote |
| 6 | **Modi 2019** | IN | 2019 | 0.605 | 0.788 | 0.697 | +0.184 | 6.33 | CCFI asymptote |
| 7 | Sunflower | TW | 2014 | 0.625 | 0.789 | 0.707 | +0.165 | — | Cross-cult. civic |
| 8 | Charlie Hebdo | FR | 2015 | 0.592 | 0.730 | 0.661 | +0.138 | 104.66 | Flash Viral |
| 9 | Chávez | VE | 2013 | 0.594 | 0.720 | 0.657 | +0.126 | 16.67 | Volatile |
| 10 | Ciolacu | RO | 2025–26 | 0.622 | 0.745 | 0.683 | +0.123 | 6.57 | Campaign |
| 11 | Macron | FR | 2017 | 0.593 | 0.714 | 0.654 | +0.121 | 12.53 | Campaign |
| 12 | **Iran-Israel** | INT | 2024 | **0.630** | 0.732 | 0.681 | +0.102 | 17.81 | Geopolitical Event |
| 13 | Georgescu | RO | 2024 | 0.595 | 0.700 | 0.648 | +0.106 | 65.33 | Flash Viral |
| 14 | **Netanyahu** | IL | 2023–24 | 0.610 | 0.698 | 0.654 | +0.087 | 7.02 | Pre-Sorted Wartime |
| 15 | Simion | RO | 2024 | 0.598 | 0.685 | 0.641 | +0.087 | 12.41 | Volatile |
| 16 | Zelensky | UA | 2022 | 0.604 | 0.660 | 0.632 | +0.056 | 5.11 | Wartime Defender |
| 17 | Orbán | HU | 2022 | 0.604 | 0.605 | 0.604 | +0.001 | 2.31 | Institutional |
| — | Chávez (acute) | VE | Mar 2013 | — | — | 0.380* | — | 16.67 | Dual-Mode |

*Estimated. All other D_new = real pipeline outputs (Jobs #10–#29).

---

## ICI Architecture — Seven Structural Levels

| Level | ICI range | Symbols | Mechanism |
|-------|-----------|---------|-----------|
| CCFI acute | **0.90+** | Modi 2014 (0.915) | No pre-established Anglo-American frameworks |
| Western ICI ceiling | **0.83–0.84** | Trump, Mandela, Putin | Electoral polarization / legacy / aggressor amplification |
| CCFI asymptote | **0.79–0.81** | Modi 2019 (0.788), Modi 2024 (0.802) | Irreducible cross-cultural incompatibility |
| Cross-cultural civic | **0.78–0.79** | Sunflower (0.789) | Partial cultural framing diversity |
| Pre-sorted moderate | **0.69–0.75** | Netanyahu, Ciolacu, Macron, Georgescu, Chávez, Iran-Israel | Institutionalized ideological sorting |
| Convergent wartime | **0.65–0.67** | Zelensky (0.660) | Crisis frame convergence |
| Institutional stable | **0.60–0.61** | Orbán (0.605) | Long-term topical stabilization |

---

## Cross-Cultural Frame Incompatibility (CCFI)

CCFI applies when Anglo-American journalism **lacks pre-established interpretive categories** for a political symbol, forcing outlets to construct incompatible frameworks from scratch.

### The CCFI Lifecycle — Three Phases (Modi 2014→2019→2024)

| Phase | Year | ICI | λ | Mechanism |
|-------|------|-----|---|-----------|
| 1. CCFI acute | 2014 | 0.915 | 2.37 | No frameworks — maximum incompatibility |
| 2. Partial decay | 2019 | 0.788 | 6.33 | Basic categories accumulate |
| 3. CCFI asymptote | 2024 | 0.802 | 9.11 | Irreducible incompatibilities persist |

> **CCFI Asymptote (~0.79–0.80):** The residual floor below which CCFI cannot decay. Reflects structural mismatches between Anglo-American journalistic categories and Non-Western political realities that no amount of coverage can resolve (secularism vs. Hindu nationalism, majoritarian democracy vs. minority rights).

> **λ inverse correlation:** As CCFI decays (ICI falls), λ rises — the symbol becomes more temporally concentrated as it becomes more familiar.

### CCFI Discrimination — Modi vs. Netanyahu

| | Modi 2014 | Netanyahu 2023 |
|---|---|---|
| ICI | 0.915 | 0.698 |
| Anglo-American framework | ABSENT | PRESENT (Israel-Palestine) |
| Result | CCFI — ceiling broken | Pre-sorted — no CCFI |

**Active war + high controversy ≠ CCFI.** Pre-established frameworks produce moderate ICI regardless of conflict intensity.

---

## Geopolitical Event Symbol — Iran-Israel 2024

First event-based symbol in the validated dataset (all others are political actors).

| PE | ICI | D_new | ICI−PE | λ |
|----|-----|-------|--------|---|
| **0.630** (max dataset) | 0.732 | 0.681 | +0.102 | 17.81 |

**Multi-Domain Geopolitical Activation (MDGA):** A direct Iran-Israel military exchange simultaneously activates nuclear, regional security, economic, diplomatic, energy, and domestic political domains — generating maximum PE within 6 months vs. years for actor symbols.

**Key finding:** PE contribution to D is competitive with ICI for the first time in the dataset — suggesting event symbols may require different α calibration than actor symbols.

---

## Dual-Mode SRM and Acute Amplification Factor (AAF)

- **SUSTAINED SRM** — long-term fragmented presence
- **ACUTE SRM** — short-term mobilization during narrative coherence
- **AAF = ACUTE / SUSTAINED**

Hugo Chávez: AAF = 9.5 (SRM 0.0121 → 0.1154 as D collapsed 0.720 → 0.380 during 11-day death window).

---

## The 16 Typological Categories

| # | Category | Example | ICI−PE | Mechanism |
|---|----------|---------|--------|-----------|
| 1 | Non-Western CCFI | Modi 2014 | +0.339 | Cross-Cultural Frame Incompatibility |
| 2 | High-Velocity Campaign | Trump | +0.293 | Electoral polarization — ICI ceiling |
| 3 | Legacy Resonance | Mandela | +0.272 | Post-mortem legacy contestation |
| 4 | Wartime Aggressor | Putin | +0.267 | Moral amplification — ICI ceiling |
| 5 | CCFI Asymptote | Modi 2024 | +0.177 | Irreducible residual incompatibility |
| 6 | CCFI Transition | Modi 2019 | +0.184 | Partial frame stabilization |
| 7 | Civic Mobilization | Sunflower | +0.165 | Cross-cultural civic diversity |
| 8 | Flash Viral / Extreme | Charlie Hebdo | +0.138 | Convergence-then-contestation |
| 9 | Revolutionary Legacy | Chávez | +0.126 | Structured ICI; AAF=9.5 |
| 10 | Post-Executive Trap | Ciolacu | +0.123 | Role transition fragmentation |
| 11 | Rapid Emergence | Macron | +0.121 | Contradictory frame coexistence |
| 12 | Geopolitical Event | Iran-Israel | +0.102 | Multi-Domain Geopolitical Activation |
| 13 | Fragmented Diffusion | Georgescu | +0.106 | Flash Viral; λ=65.33 |
| 14 | Pre-Sorted Wartime | Netanyahu | +0.087 | Institutionalized conflict frames |
| 15 | Electorally Volatile | Simion | +0.087 | Affective Deficit (A=0.099) |
| 16 | Wartime Defender | Zelensky | +0.056 | Crisis frame convergence |
| 17 | Longevity Saturation | Orbán | +0.001 | Perfectly balanced PE/ICI |

---

## Comparative Dataset — Full SRM Variables

| Symbol / Context | V | A | D_new | N | SRM (λ=2) | λ | SRM (λ_emp) | Typology |
|------------------|---|---|-------|---|-----------|---|------------|---------|
| **Modi 2014 (IN)** | TBD | TBD | **0.746** | TBD | TBD | **2.37** | TBD | CCFI acute |
| **Modi 2019 (IN)** | TBD | TBD | **0.697** | TBD | TBD | **6.33** | TBD | CCFI asymptote |
| **Modi 2024 (IN)** | TBD | TBD | **0.713** | TBD | TBD | **9.11** | TBD | CCFI asymptote |
| **Netanyahu (IL)** | TBD | TBD | **0.654** | TBD | TBD | **7.02** | TBD | Pre-Sorted Wartime |
| **Iran-Israel (INT)** | TBD | TBD | **0.681** | TBD | TBD | **17.81** | TBD | Geopolitical Event |
| George Simion (RO, 2024) | 0.279 | 0.099 | 0.641 | 0.996 | 0.0054 | 12.41 | — | Electorally Volatile |
| Viktor Orbán (HU, 2022) | 0.168 | 0.236 | 0.604 | 0.812 | 0.0065 | 2.31 | 0.0051 | Longevity Saturation |
| Nelson Mandela (ZA, 2013) | 0.311 | 0.246 | 0.700 | 0.510 | 0.0088 | 19.66 | — | Legacy Resonance |
| Vladimir Putin (RU, 2022) | 0.217 | 0.259 | 0.700 | 1.000 | 0.0103 | 4.90 | 0.0009 | Wartime Aggressor |
| Hugo Chávez SUSTAINED (VE) | 0.186 | 0.290 | 0.657 | 0.941 | 0.0121 | 16.67 | — | Revolutionary Legacy |
| Emmanuel Macron (FR, 2017) | 0.507 | 0.168 | 0.654 | 1.000 | 0.0169 | 12.53 | — | Rapid Emergence |
| Călin Georgescu (RO, 2024) | 0.750 | 0.398 | 0.648 | 0.600 | 0.0307 | 65.33 | — | Flash Viral |
| Marcel Ciolacu (RO, 2026) | 0.720 | 0.420 | 0.683 | 0.650 | 0.0365 | 6.57 | 0.0008 | Post-Executive Trap |
| Sunflower Movement (TW, 2014) | 0.680 | 0.420 | 0.707 | 0.580 | 0.0376 | — | — | Civic Mobilization |
| Charlie Hebdo (FR, 2015) | TBD | TBD | 0.661 | TBD | TBD | 104.66 | — | Extreme Flash Viral |
| Donald Trump (US, 2016) | 0.958 | 0.580 | 0.689 | 0.720 | 0.0922 | 7.01 | 0.0023 | High-Velocity Campaign |
| Hugo Chávez ACUTE (VE, 2013) | 0.689 | 0.358 | 0.380* | 1.000 | 0.1154 | 16.67 | — | Dual-Mode |
| Volodymyr Zelensky (UA, 2022) | 0.873 | 0.640 | 0.632 | 0.781 | 0.1121 | 5.11 | 0.0135 | Wartime Defender |

*D_acute estimated. Modi, Netanyahu, Iran-Israel: V, A, N pending full pipeline.

---

## SSRN Publications

All papers: [10.17605/OSF.IO/HYDNZ](https://doi.org/10.17605/OSF.IO/HYDNZ)

| Paper | Job | Key finding |
|-------|-----|-------------|
| Trump v2 | #10 | ICI ceiling rank 1 Western (+0.293) |
| Ciolacu v2 | #12 | Romanian Triad · PE convergence |
| Georgescu v2 | #14 | Largest D discrepancy (−26.5%) |
| Sunflower v2 | #15 | Highest D_new (0.707) |
| Orbán v2 | #16 | ICI−PE≈0 · Longevity Paradox |
| Zelensky v2 | #17 | Wartime defender · H3 reformulated |
| Putin v2 | #18 | Antagonistic Pair · ICI≈Trump |
| Simion v2 | #19 | Affective Deficit (A=0.099) |
| Mandela v2 | #20 | Legacy Contestation ICI |
| Macron v2 | #21 | Rapid Emergence Paradox |
| Chávez v2 | #22 | Dual-Mode SRM · AAF=9.5 |
| Charlie Hebdo | #24 | Flash Viral rank 1 · λ=104.66 |
| **Modi 2014** ★ | **#25** | **ICI=0.915 · CCFI · λ=2.37** |
| **Netanyahu** ★ | **#26** | **ICI=0.698 · Pre-sorted · λ=7.02** |
| **Modi 2019** ★ | **#27** | **CCFI decay · λ=6.33** |
| **Modi 2024** ★ | **#28** | **CCFI asymptote · λ=9.11** |
| **Iran-Israel** ★ | **#29** | **PE max (0.630) · MDGA · λ=17.81** |
| **Synthetic v3** ★ | — | **18 symbol-periods · CCFI lifecycle · 10 findings** |

---

## Repository Structure

```
Politomorphism/
├── .github/workflows/
│   ├── srm_compute_D.yml              ← PE/ICI pipeline (Jobs #10–#29)
│   └── fetch_trends.yml              ← λ calibration
├── scripts/
│   ├── get_trends.py                  ← λ calibration (Step 0)
│   ├── compute_D.py                   ← D = α·PE + (1−α)·ICI
│   ├── calibrate_alpha.py             ← α_optimal = 0.338
│   └── test_hypotheses.py             ← H1/H2/H3/H4
├── docs/
│   ├── srm_pe_ici_map.svg
│   └── srm_pe_ici_map_interactive.html
├── results/
│   └── D_result_*.json
└── README.md
```

---

## Reproducibility

- **Data:** [mediacloud.org](https://mediacloud.org) + [Google Trends](https://trends.google.com)
- **λ calibration:** `scipy.optimize.brentq`, Python 3.11, GitHub Actions ubuntu-latest
- **Sentiment:** VADER (`vaderSentiment 3.3.2`) English; DistilBERT Romanian
- **D computation:** LDA (scikit-learn K=10 seed=42) + `paraphrase-multilingual-MiniLM-L12-v2`
- **α_optimal = 0.338**, 14 real-value entries
- **Bootstrap:** n=20 (speed); publication requires n≥200

---

## Preregistration

OSF: [10.17605/OSF.IO/HYDNZ](https://doi.org/10.17605/OSF.IO/HYDNZ)  
Zenodo: [10.5281/zenodo.18962821](https://doi.org/10.5281/zenodo.18962821)

---

## Citation

```bibtex
@misc{serban2026politomorphism,
  author  = {Serban, Gabriel Florin},
  title   = {Politomorphism: Social Resonance Model —
             PE/ICI Decomposition across 18 Validated Symbol-Periods},
  year    = {2026},
  doi     = {10.17605/OSF.IO/HYDNZ},
  url     = {https://github.com/profserbangabriel-del/Politomorphism},
  license = {CC BY 4.0}
}
```

---

*Politomorphism Research Project · Serban Gabriel Florin · Romania / EU · March 2026*
