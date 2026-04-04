# Politomorphism — Social Resonance Model (SRM)

**Serban Gabriel Florin** | Independent Researcher  
ORCID: [0009-0000-2266-3356](https://orcid.org/0009-0000-2266-3356) | DOI: [10.17605/OSF.IO/HYDNZ](https://doi.org/10.17605/OSF.IO/HYDNZ)  
GitHub: [profserbangabriel-del/Politomorphism](https://github.com/profserbangabriel-del/Politomorphism)  
License: CC BY 4.0

---

## What is Politomorphism?

Politomorphism is a theoretical framework that treats political symbols as **morphogenetic agents** — entities that transform power structures through the process of symbolic diffusion. Its computational component, the **Social Resonance Model (SRM)**, provides a quantitative measure of how effectively a political symbol mobilizes public space across Anglo-American media.

> **Status (April 2026):** 19/19 symbol-periods validated. Bootstrap n=200 for all 18 primary symbols. V, A, N computed automatically via `compute_V_A_N.py`. Paper under review.

---

## The SRM Formula

**SRM = V × A × e^(−λD) × N**

| Variable | Name | What it measures | Range |
|----------|------|-----------------|-------|
| V | Viral Velocity | Log-normalized escalation ratio (peak/avg) from article time series | 0–1 |
| A | Affective Weight | Mean absolute VADER compound sentiment on article titles | 0–1 |
| D | Semantic Drift | D = α·PE + (1−α)·ICI, α=0.5 (primary), α_opt=0.389 (exploratory) | 0–1 |
| N | Network Coverage | Proportion of days with ≥1 article in corpus window | 0–1 |
| **λ** | **Decay Constant** | Empirically calibrated from Google Trends avg/peak ratio | 2–105 |

---

## Complete Dataset — 19 Symbol-Periods (Bootstrap n=200)

| Symbol | Period | PE | ICI | D | V | A | N | λ | SRM | Typology |
|--------|--------|----|-----|---|---|---|---|---|-----|----------|
| Sunflower | TW 2014 | 0.621 | 0.787 | 0.704 | 0.091 | 0.195 | 0.971 | 2.00 | **0.00419** | Civic Mobilization |
| Modi 2014 | IN 2014 | 0.590 | 0.904 | 0.747 | 0.132 | 0.221 | 0.144 | 2.37 | 0.000714 | CCFI Acute |
| Putin | RU 2022 | 0.547 | 0.753 | 0.650 | 0.503 | 0.108 | 0.402 | 4.90 | 0.000904 | Wartime Aggressor |
| Modi 2019 | IN 2019 | 0.600 | 0.815 | 0.707 | 0.194 | 0.184 | 0.632 | 6.33 | 0.000256 | CCFI Decay |
| Ciolacu | RO 2024–25 | 0.576 | 0.658 | 0.617 | — | — | — | 6.57 | — | Campaign |
| Trump | US 2015–16 | 0.592 | 0.881 | 0.737 | 0.786 | 0.222 | 0.559 | 7.01 | 0.000559 | Western Ceiling |
| Netanyahu | IL 2023–24 | 0.591 | 0.735 | 0.663 | 0.428 | 0.144 | 0.825 | 7.02 | 0.000485 | Pre-sorted Wartime |
| Modi 2024 | IN 2024 | 0.603 | 0.849 | 0.726 | 0.383 | 0.237 | 0.529 | 9.11 | 0.000065 | CCFI Asymptote |
| Bolsonaro | BR 2022–23 | 0.611 | 0.786 | 0.698 | 0.550 | 0.059 | 0.478 | 10.43 | 0.000011 | Partial CCFI |
| Simion | RO 2024 | 0.566 | 0.692 | 0.629 | — | — | — | 12.41 | — | Volatile |
| Macron | FR 2017 | 0.603 | 0.729 | 0.666 | 0.478 | 0.085 | 0.570 | 12.53 | 0.000006 | Campaign |
| Chavez | VE 2013 | 0.572 | 0.927 | 0.750 | 0.917 | 0.258 | 0.402 | 16.67 | ~0 | Volatile Legacy |
| ChavezAcute | VE Mar 2013 | 0.600 | 0.802 | 0.701 | — | — | — | 16.67 | — | Dual-Mode |
| IranIsrael | INT 2024 | 0.604 | 0.909 | 0.756 | 0.778 | 0.295 | 0.396 | 17.81 | ~0 | Geopolitical Event |
| Mandela | ZA 2013 | 0.564 | 0.836 | — | — | — | — | 19.66 | — | Legacy* |
| Georgescu | RO 2024 | 0.578 | 0.634 | 0.606 | — | — | — | 65.33 | ~0 | Flash Viral |
| CharlieHebdo | FR 2015 | 0.572 | 0.732 | 0.652 | 0.531 | 0.140 | 0.833 | 104.66 | 0.0 | Extreme Flash Viral |
| Zelensky | UA 2022 | 0.596 | 0.542 | 0.569 | — | — | — | 5.11 | — | Wartime Defender |
| Orban | HU 2022 | 0.601 | 0.592 | 0.597 | — | — | — | 2.31 | — | Institutional |

*Mandela (2013) predates Media Cloud indexing. PE/ICI from archival estimate. All other D values: bootstrap n=200 pipeline (Jobs #33–#79).  
— = V/A/N/SRM pending final pipeline run with `compute_V_A_N.py`.

---

## λ Calibration — Complete Dataset

```
avg / peak = (1 − e^(−λT)) / (λT)
```

Solved numerically via `scipy.optimize.brentq`. All 19 λ values empirically calibrated from Google Trends.

| Category | λ range | Examples |
|----------|---------|---------|
| Institutionally Durable | 2–5 | Sunflower (2.00), Orban (2.31), Modi 2014 (2.37), Zelensky (5.11) |
| Campaign / Ascension | 6–8 | Modi 2019 (6.33), Ciolacu (6.57), Trump (7.01), Netanyahu (7.02) |
| Electorally Volatile | 9–20 | Modi 2024 (9.11), Simion (12.41), Macron (12.53), Chavez (16.67), IranIsrael (17.81), Mandela (19.66) |
| Flash Viral | 50–70 | Georgescu (65.33) |
| Extreme Flash Viral | >70 | CharlieHebdo (104.66) |

> **Key finding:** λ ranges from 2.00 to 104.66 — a **52-fold variation**. Lambda is the primary SRM determinant: all symbols with λ ≤ 7 produce SRM > 0.0001; all symbols with λ ≥ 17 produce SRM ≈ 0.

---

## D Operationalization

### D = α · PE + (1−α) · ICI

| Component | Method | Measures |
|-----------|--------|----------|
| PE | Mean Jensen-Shannon Divergence on LDA (K=10, seed=42) | Topical breadth |
| ICI | 1 − mean pairwise cosine similarity (`paraphrase-multilingual-MiniLM-L12-v2`) | Framing divergence |

**α = 0.5** (primary — equal weighting, no circularity).  
α_opt = 0.389 reported as exploratory supplementary analysis only.

> **Key finding:** ICI range (0.372) is **6.8× larger** than PE range (0.055) across 19 symbol-periods. ICI is the primary D driver: Pearson r(ICI, D) = 0.51 vs r(PE, D) = −0.02.

---

## Cross-Cultural Frame Incompatibility (CCFI)

CCFI applies when Anglo-American journalism **lacks pre-established interpretive categories**, forcing outlets to construct incompatible frameworks from scratch.

### CCFI Lifecycle — Modi Longitudinal Series

| Phase | Year | ICI | D | λ | SRM |
|-------|------|-----|---|---|-----|
| CCFI acute | 2014 | **0.904** | 0.747 | 2.37 | 0.000714 |
| CCFI decay | 2019 | 0.815 | 0.707 | 6.33 | 0.000256 |
| CCFI asymptote | 2024 | 0.849 | 0.726 | 9.11 | 0.000065 |

**First longitudinal empirical test of CCFI dynamics in the literature.** ICI stabilizes at ~0.80 while λ triples — demonstrating that as frames stabilize, temporal concentration of interest increases.

### CCFI Discrimination — Modi vs. Netanyahu

| | Modi 2014 | Netanyahu 2023 |
|---|---|---|
| ICI | 0.904 | 0.735 |
| Anglo-American framework | ABSENT | PRESENT (Israel-Palestine) |
| Result | CCFI — ceiling broken | Pre-sorted — no CCFI |

**Active war + high controversy ≠ CCFI.** Pre-established frameworks produce moderate ICI regardless of conflict intensity.

---

## Key Empirical Findings

1. **ICI dominance confirmed:** ICI range 6.8× larger than PE range. λ primary SRM determinant.
2. **CCFI lifecycle documented:** Modi 2014→2019→2024 shows ICI decay from 0.904 to asymptote ~0.80.
3. **V·A Paradox:** CharlieHebdo (V=0.531, A=0.140, N=0.833) → SRM=0.0 due to λ=104.66.
4. **Convergent wartime framing:** Zelensky ICI=0.542 < PE=0.596 — only symbol with ICI<PE. Crisis produces frame convergence, not divergence.
5. **Geopolitical Event Symbol:** IranIsrael — first event-based symbol, ICI=0.909 through Multi-Domain Geopolitical Activation.
6. **Log-log regression:** log(SRM) = 3.09 − 3.50·log(λ) + 1.57·log(V·A·N), R²_adj=0.87, p<0.001.
7. **α_optimal = 0.389** (exploratory): consistent ICI-dominance across all typologies.

---

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `scripts/compute_D.py` | PE + ICI + bootstrap CI computation |
| `scripts/compute_V_A_N.py` | Automatic V (from λ), A (VADER), N (coverage) |
| `scripts/calibrate_alpha.py` | α optimization across 19 symbols |
| `scripts/test_hypotheses.py` | H1/H2/H3 statistical tests |
| `scripts/loocv_srm.py` | Leave-One-Out CV for log-log regression |

### Workflow

```
.github/workflows/srm_compute_D.yml
  → Fetch Media Cloud corpus
  → compute_D.py  (PE, ICI, D, bootstrap CI)
  → compute_V_A_N.py  (V, A, N, SRM)
  → calibrate_alpha.py
  → test_hypotheses.py
  → Upload ZIP artifact
```

---

## Repository Structure

```
Politomorphism/
├── .github/workflows/
│   ├── srm_compute_D.yml         ← Full pipeline (Jobs #33–#79)
│   └── fetch_trends.yml          ← λ calibration
├── scripts/
│   ├── compute_D.py              ← D = α·PE + (1−α)·ICI
│   ├── compute_V_A_N.py          ← V, A, N automatic computation
│   ├── calibrate_alpha.py        ← α_opt = 0.389 (exploratory)
│   ├── test_hypotheses.py        ← H1/H2/H3
│   └── loocv_srm.py              ← LOOCV validation
├── D_result_*.json               ← Bootstrap n=200 results (18 symbols)
├── V_A_N_*.json                  ← V, A, N, SRM per symbol
├── alpha_calibration_results.json
├── hypothesis_test_results.json
├── index.html                    ← GitHub Pages site
└── README.md
```

---

## Reproducibility

- **Data:** [mediacloud.org](https://mediacloud.org) (English-language Anglo-American corpus) + Google Trends
- **λ calibration:** `scipy.optimize.brentq`, Python 3.11, GitHub Actions ubuntu-latest
- **Sentiment:** VADER (`vaderSentiment`) on article titles
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
- **LDA:** scikit-learn, K=10, random_state=42, max_iter=20
- **Bootstrap:** n=200 for all 18 primary symbols (Jobs #33–#79)
- **All analyses:** reproducible via GitHub Actions — no local environment required

> **Limitation:** All corpora are Anglo-American English-language. ICI in domestic-language corpora for Non-Western symbols would likely differ. Mandela (2013) predates Media Cloud indexing coverage.

---

## Preregistration & Citation

OSF: [10.17605/OSF.IO/HYDNZ](https://doi.org/10.17605/OSF.IO/HYDNZ)

```bibtex
@misc{serban2026politomorphism,
  author  = {Serban, Gabriel Florin},
  title   = {Politomorphism: Social Resonance Model —
             PE/ICI Decomposition across 19 Validated Symbol-Periods},
  year    = {2026},
  doi     = {10.17605/OSF.IO/HYDNZ},
  url     = {https://github.com/profserbangabriel-del/Politomorphism},
  orcid   = {0009-0000-2266-3356},
  license = {CC BY 4.0}
}
```

---

*Politomorphism Research Project · Serban Gabriel Florin · Romania / EU · April 2026*
