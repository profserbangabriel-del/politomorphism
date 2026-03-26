"""
SRM – Empirical Calibration of α Weighting Parameter
PE/ICI Retrovalidation on All Available Cases

Serban Gabriel Florin | Politomorphism Research Project
GitHub: profserbangabriel-del/politomorphism
ORCID: 0009-0000-2266-3356

Scope:
  1. Computes PE and ICI for all 4 pipeline-available symbols
     (Georgescu, Ciolacu, Trump, Sunflower Movement)
  2. Runs bootstrap n=500 on each symbol for publishable CI
  3. Regresses D_legacy ~ α·PE + (1−α)·ICI to find empirical α
  4. Writes results/D_alpha_calibration.json and results/D_alpha_calibration.md

Usage (local):
  python scripts/compute_D_alpha_calibration.py

Usage (GitHub Actions):
  Triggered by workflow .github/workflows/srm_alpha_calibration.yml
  Input CSVs expected in: data/{symbol_key}/articles.csv  (column: title)
  D_legacy values are hardcoded from validated publications.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ALPHA_DEFAULT = 0.5
BOOTSTRAP_N = 500          # publishable CI (≥200 required; 500 recommended)
ICI_SAMPLE_SIZE = 3000     # max titles sampled for O(n²) cosine matrix
LDA_TOPICS = 10
LDA_RANDOM_STATE = 42
RANDOM_SEED = 42

# Symbol registry
# D_legacy: implicit values from prior validated SRM publications
# data_file: relative path from repo root to CSV with column 'title'
SYMBOLS = {
    "georgescu": {
        "label": "Călin Georgescu (RO, 2024)",
        "D_legacy": 0.881,
        "lambda_emp": 65.33,
        "V": 0.750, "A": 0.398, "N": 0.600,
        "data_file": "data/georgescu/articles.csv",
    },
    "ciolacu": {
        "label": "Marcel Ciolacu (RO, 2025–26)",
        "D_legacy": 0.841,
        "lambda_emp": 6.57,
        "V": 0.720, "A": 0.420, "N": 0.650,
        "data_file": "data/ciolacu/articles.csv",
    },
    "trump": {
        "label": "Donald Trump (US, 2015–16)",
        "D_legacy": 0.734,
        "lambda_emp": 7.01,
        "V": 0.958, "A": 0.580, "N": 0.720,
        "data_file": "data/trump/articles.csv",
    },
    "sunflower": {
        "label": "Sunflower Movement (TW, 2014)",
        "D_legacy": 0.774,
        "lambda_emp": None,   # GT resolution insufficient
        "V": 0.680, "A": 0.393, "N": 0.580,
        "data_file": "data/sunflower/articles.csv",
    },
}

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def load_titles(csv_path: str) -> list[str]:
    """Load article titles from CSV. Drops NaN and empty strings."""
    df = pd.read_csv(csv_path, usecols=["title"])
    titles = df["title"].dropna().astype(str).str.strip()
    titles = titles[titles.str.len() > 5].tolist()
    return titles


def compute_PE(titles: list[str]) -> float:
    """
    Polysemy Entropy via LDA + Jensen-Shannon Divergence.
    Returns PE ∈ [0, 1].
    """
    if len(titles) < 20:
        raise ValueError(f"Corpus too small for PE: {len(titles)} titles")

    vec = TfidfVectorizer(
        max_features=5000,
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vec.fit_transform(titles)

    lda = LatentDirichletAllocation(
        n_components=LDA_TOPICS,
        random_state=LDA_RANDOM_STATE,
        max_iter=20,
        learning_method="batch",
    )
    topic_dist = lda.fit_transform(tfidf_matrix)

    # Laplace smoothing + renormalize
    topic_dist = topic_dist + 1e-10
    topic_dist = topic_dist / topic_dist.sum(axis=1, keepdims=True)

    centroid = topic_dist.mean(axis=0)
    centroid = centroid / centroid.sum()

    pe = float(np.mean([jensenshannon(d, centroid, base=2) for d in topic_dist]))
    return pe


def compute_ICI(titles: list[str], sample_size: int = ICI_SAMPLE_SIZE,
                model: SentenceTransformer = None) -> float:
    """
    Intra-contextual Incoherence via sentence embeddings + mean pairwise cosine distance.
    Returns ICI ∈ [0, 1].
    Samples corpus if len(titles) > sample_size to avoid OOM on O(n²) matrix.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    if len(titles) > sample_size:
        idx = rng.choice(len(titles), size=sample_size, replace=False)
        sampled = [titles[i] for i in idx]
        sample_fraction = sample_size / len(titles)
    else:
        sampled = titles
        sample_fraction = 1.0

    if model is None:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    embeddings = model.encode(
        sampled,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )  # shape: (n, 384), L2-normalized

    # Chunked cosine similarity to avoid single large allocation
    n = len(embeddings)
    chunk = 500
    total_sim = 0.0
    count = 0
    for i in range(0, n, chunk):
        batch = embeddings[i : i + chunk]
        sim_block = batch @ embeddings.T   # (chunk, n)
        total_sim += sim_block.sum()
        count += sim_block.shape[0] * sim_block.shape[1]

    mean_cos_sim = total_sim / count
    ici = float(1.0 - mean_cos_sim)

    return ici, sample_fraction


def bootstrap_D(titles: list[str], model: SentenceTransformer,
                alpha: float = ALPHA_DEFAULT, n_iter: int = BOOTSTRAP_N) -> dict:
    """
    Bootstrap CI for D = α·PE + (1−α)·ICI.
    Returns dict with mean, std, ci_low, ci_high, iterations.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    D_samples = []

    for i in range(n_iter):
        idx = rng.choice(len(titles), size=len(titles), replace=True)
        sample = [titles[j] for j in idx]
        try:
            pe_b = compute_PE(sample)
            ici_b, _ = compute_ICI(sample, model=model)
            D_samples.append(alpha * pe_b + (1 - alpha) * ici_b)
        except Exception:
            continue  # skip failed resamples

    D_samples = np.array(D_samples)
    std = float(np.std(D_samples))
    return {
        "mean": float(np.mean(D_samples)),
        "std": std,
        "ci_low": float(np.percentile(D_samples, 2.5)),
        "ci_high": float(np.percentile(D_samples, 97.5)),
        "n_valid": len(D_samples),
    }


def calibrate_alpha(results: dict) -> dict:
    """
    Empirical calibration of α via OLS minimisation.
    Minimises Σ (D_legacy_i − [α·PE_i + (1−α)·ICI_i])² over α ∈ [0, 1].

    Returns dict with alpha_empirical, residuals, R2, and interpretation.
    """
    valid = {k: v for k, v in results.items()
             if "PE" in v and "ICI" in v and v["PE"] is not None}

    if len(valid) < 2:
        return {"error": "Insufficient data for α calibration (need ≥2 symbols)"}

    D_legacy = np.array([v["D_legacy"] for v in valid.values()])
    PE_vals   = np.array([v["PE"]       for v in valid.values()])
    ICI_vals  = np.array([v["ICI"]      for v in valid.values()])

    def sse(alpha):
        D_pred = alpha * PE_vals + (1 - alpha) * ICI_vals
        return float(np.sum((D_legacy - D_pred) ** 2))

    opt = minimize_scalar(sse, bounds=(0.0, 1.0), method="bounded")
    alpha_emp = float(opt.x)

    D_pred = alpha_emp * PE_vals + (1 - alpha_emp) * ICI_vals
    residuals = D_legacy - D_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((D_legacy - D_legacy.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Interpretation
    if abs(alpha_emp - 0.5) < 0.05:
        interpretation = "α_emp ≈ 0.5 — prior neutral confirmat empiric"
    elif alpha_emp > 0.5:
        interpretation = f"α_emp={alpha_emp:.3f} — PE domină ICI; topical breadth ponderată mai sus"
    else:
        interpretation = f"α_emp={alpha_emp:.3f} — ICI domină PE; framing divergence ponderată mai sus"

    return {
        "alpha_empirical": alpha_emp,
        "alpha_prior": ALPHA_DEFAULT,
        "delta_alpha": alpha_emp - ALPHA_DEFAULT,
        "SSE": ss_res,
        "R2": r2,
        "residuals": {k: float(r) for k, r in zip(valid.keys(), residuals)},
        "interpretation": interpretation,
        "n_symbols": len(valid),
    }


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SRM α Calibration Pipeline")
    print(f"Bootstrap iterations: {BOOTSTRAP_N}")
    print("=" * 60)

    # Load sentence transformer once (shared across all symbols)
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("Model loaded.")

    results = {}

    for symbol_key, cfg in SYMBOLS.items():
        print(f"\n{'─'*50}")
        print(f"Symbol: {cfg['label']}")
        data_path = repo_root / cfg["data_file"]

        if not data_path.exists():
            print(f"  ⚠  Data file not found: {data_path}")
            print(f"     Skipping — add {cfg['data_file']} to enable.")
            results[symbol_key] = {
                "label": cfg["label"],
                "D_legacy": cfg["D_legacy"],
                "PE": None, "ICI": None, "D_new": None,
                "status": "DATA_MISSING",
            }
            continue

        t0 = time.time()
        titles = load_titles(str(data_path))
        print(f"  Loaded {len(titles):,} titles from {data_path.name}")

        # PE
        print("  Computing PE (LDA)...")
        pe = compute_PE(titles)
        print(f"  PE = {pe:.4f}")

        # ICI
        print("  Computing ICI (embeddings)...")
        ici, sample_frac = compute_ICI(titles, model=model)
        print(f"  ICI = {ici:.4f}  (sample fraction: {sample_frac:.1%})")

        # D composite
        D_new = ALPHA_DEFAULT * pe + (1 - ALPHA_DEFAULT) * ici
        D_delta = D_new - cfg["D_legacy"]
        print(f"  D_new (α=0.5) = {D_new:.4f}  |  D_legacy = {cfg['D_legacy']}  |  Δ = {D_delta:+.4f} ({D_delta/cfg['D_legacy']*100:+.1f}%)")

        # Bootstrap
        print(f"  Bootstrap n={BOOTSTRAP_N}...")
        boot = bootstrap_D(titles, model=model, alpha=ALPHA_DEFAULT, n_iter=BOOTSTRAP_N)
        print(f"  Bootstrap: mean={boot['mean']:.4f}  std={boot['std']:.4f}  95%CI=[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]  n_valid={boot['n_valid']}")

        elapsed = time.time() - t0
        print(f"  ✓ Done in {elapsed/60:.1f} min")

        results[symbol_key] = {
            "label": cfg["label"],
            "D_legacy": cfg["D_legacy"],
            "PE": pe,
            "ICI": ici,
            "ICI_sample_fraction": sample_frac,
            "D_new_alpha05": D_new,
            "D_delta_abs": D_delta,
            "D_delta_pct": D_delta / cfg["D_legacy"] * 100,
            "bootstrap": boot,
            "V": cfg["V"], "A": cfg["A"], "N": cfg["N"],
            "lambda_emp": cfg["lambda_emp"],
            "status": "OK",
        }

    # ── α Calibration ──
    print(f"\n{'─'*50}")
    print("Calibrating α...")
    alpha_result = calibrate_alpha(results)
    print(f"  α_empirical = {alpha_result.get('alpha_empirical', 'N/A'):.4f}")
    print(f"  Interpretation: {alpha_result.get('interpretation', 'N/A')}")

    # Recompute D_new with empirical alpha where available
    alpha_emp = alpha_result.get("alpha_empirical", ALPHA_DEFAULT)
    for sym, r in results.items():
        if r["PE"] is not None:
            r["D_new_alpha_emp"] = alpha_emp * r["PE"] + (1 - alpha_emp) * r["ICI"]
            r["D_delta_alpha_emp"] = r["D_new_alpha_emp"] - r["D_legacy"]

    # ── Save JSON ──
    output = {
        "meta": {
            "pipeline": "compute_D_alpha_calibration.py",
            "bootstrap_n": BOOTSTRAP_N,
            "alpha_default": ALPHA_DEFAULT,
            "ici_sample_size": ICI_SAMPLE_SIZE,
            "lda_topics": LDA_TOPICS,
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "alpha_calibration": alpha_result,
        "symbols": results,
    }

    json_path = results_dir / "D_alpha_calibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON saved: {json_path}")

    # ── Save Markdown report ──
    md_lines = [
        "# SRM α Calibration — Results\n",
        f"*Generated: {output['meta']['timestamp']}*\n",
        "## α Empirical Calibration\n",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| α default (prior) | {ALPHA_DEFAULT} |",
        f"| α empirical | {alpha_result.get('alpha_empirical', 'N/A'):.4f} |",
        f"| Δα | {alpha_result.get('delta_alpha', 'N/A'):.4f} |",
        f"| R² | {alpha_result.get('R2', 'N/A'):.4f} |",
        f"| SSE | {alpha_result.get('SSE', 'N/A'):.6f} |",
        f"| n symbols | {alpha_result.get('n_symbols', 'N/A')} |",
        f"\n**Interpretation:** {alpha_result.get('interpretation', 'N/A')}\n",
        "\n## Per-Symbol Results\n",
        "| Symbol | D_legacy | PE | ICI | D_new(α=0.5) | D_new(α_emp) | Δ_legacy | 95% CI | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sym, r in results.items():
        if r["status"] == "DATA_MISSING":
            md_lines.append(f"| {r['label']} | {r['D_legacy']} | — | — | — | — | — | — | ⚠ MISSING |")
        else:
            ci = r["bootstrap"]
            md_lines.append(
                f"| {r['label']} | {r['D_legacy']} | {r['PE']:.4f} | {r['ICI']:.4f} | "
                f"{r['D_new_alpha05']:.4f} | {r.get('D_new_alpha_emp', float('nan')):.4f} | "
                f"{r['D_delta_abs']:+.4f} ({r['D_delta_pct']:+.1f}%) | "
                f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] | ✓ |"
            )

    md_path = results_dir / "D_alpha_calibration.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"✓ Markdown report saved: {md_path}")

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok_count = sum(1 for r in results.values() if r["status"] == "OK")
    print(f"Symbols processed: {ok_count}/{len(SYMBOLS)}")
    if "alpha_empirical" in alpha_result:
        print(f"α_empirical = {alpha_result['alpha_empirical']:.4f}  (prior = {ALPHA_DEFAULT})")
        print(f"R² = {alpha_result['R2']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
