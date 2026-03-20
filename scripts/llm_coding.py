"""
llm_coding.py
Politomorphism Research Project | Serban Gabriel Florin
License: CC BY 4.0

Automated coding of PE (topical diversity) and Framing (political tone)
for the Trump validation subset using Claude API.

Based on: Gilardi et al. (2023) "ChatGPT outperforms crowd-workers for
text-annotation tasks" PNAS — LLM coding is acceptable when validated
against human judgments on a subset.

Usage:
    python llm_coding.py --input trump_validation_sample.csv --output llm_coded.csv

Then validate on 50 articles manually and run:
    python llm_coding.py --step correlate --llm llm_coded.csv --human human_coded_50.csv
"""

import pandas as pd
import numpy as np
import json
import time
import argparse
import os
import re
from pathlib import Path

# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a political communication researcher coding news articles.
You will receive a news article title (and URL for context).
You must assign TWO scores and return ONLY valid JSON, nothing else.

SCORE 1 — PE (Topical Diversity), scale 1-5:
1 = Single topic, very focused (e.g., only about immigration)
2 = One main topic + one secondary reference
3 = Two distinct topics, roughly balanced
4 = Three or more topics, no clear dominance
5 = Highly diverse, mix of unrelated topics

Use these 10 predefined topics:
T01: Electoral campaign / polling / strategy
T02: Immigration / border / deportation
T03: Economy / trade / jobs / taxes
T04: Foreign policy / NATO / Russia / China
T05: Media / press / Twitter / social media
T06: Republican Party / primaries / debates
T07: Legal / judicial / investigations / ethics
T08: Race / religion / discrimination / social issues
T09: Clinton / Democratic Party / general election
T10: Personal / family / biography / style

SCORE 2 — FRAMING (Political tone toward Trump), scale -2 to +2:
-2 = Strongly anti-Trump (harsh criticism, negative framing)
-1 = Mildly anti-Trump (nuanced critique, not hostile)
 0 = Neutral / descriptive (factual reporting, no clear evaluation)
+1 = Mildly pro-Trump (favorable tone, justifications)
+2 = Strongly pro-Trump (apologetic, strongly positive framing)

Return ONLY this JSON structure, no explanation:
{
  "PE": <integer 1-5>,
  "framing": <integer -2, -1, 0, 1, or 2>,
  "PE_topics": ["T01", "T02"],
  "confidence": "high" | "medium" | "low",
  "reasoning": "<one sentence>"
}"""

USER_TEMPLATE = """Article title: {title}
Outlet: {outlet}
Date: {date}
Phase: {phase}

Code this article."""


# ── API call ──────────────────────────────────────────────────────────────────

def code_article(title, outlet, date, phase, max_retries=3):
    """
    Calls Claude API to code a single article.
    Returns dict with PE, framing, confidence, reasoning.
    """
    import urllib.request
    import urllib.error

    user_msg = USER_TEMPLATE.format(
        title=title, outlet=outlet, date=date, phase=phase
    )

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}]
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["content"][0]["text"].strip()

                # Parse JSON from response
                # Sometimes model wraps in ```json ... ```
                text = re.sub(r"```json\s*", "", text)
                text = re.sub(r"```\s*", "", text)
                result = json.loads(text)

                # Validate fields
                assert "PE" in result and "framing" in result
                assert 1 <= int(result["PE"]) <= 5
                assert -2 <= int(result["framing"]) <= 2

                return {
                    "PE_llm": int(result["PE"]),
                    "framing_llm": int(result["framing"]),
                    "PE_topics_llm": ",".join(result.get("PE_topics", [])),
                    "confidence_llm": result.get("confidence", "medium"),
                    "reasoning_llm": result.get("reasoning", ""),
                    "error": None
                }

        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limit
                wait = (attempt + 1) * 30
                print(f"    Rate limit — waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 529:  # Overloaded
                wait = (attempt + 1) * 15
                print(f"    API overloaded — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP error {e.code}: {e.reason}")
                break

        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            print(f"    Parse error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

        except Exception as e:
            print(f"    Unexpected error: {e}")
            break

    return {
        "PE_llm": None, "framing_llm": None,
        "PE_topics_llm": None, "confidence_llm": None,
        "reasoning_llm": None, "error": "FAILED"
    }


# ── Main coding loop ──────────────────────────────────────────────────────────

def run_coding(input_csv, output_csv, resume=True, delay=0.5):
    """
    Codes all articles in input_csv using Claude API.
    Supports resuming from partial results.
    """
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} articles from {input_csv}")

    # Resume from existing output
    coded_ids = set()
    if resume and os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        coded_ids = set(existing["id"].tolist())
        print(f"Resuming — already coded: {len(coded_ids)}")

    results = []
    to_code = df[~df["id"].isin(coded_ids)].copy()
    print(f"Articles to code: {len(to_code)}")

    if len(to_code) == 0:
        print("All articles already coded.")
        return pd.read_csv(output_csv)

    for i, (_, row) in enumerate(to_code.iterrows()):
        print(f"[{i+1}/{len(to_code)}] {row['media_name']} | {str(row['title'])[:60]}...")

        result = code_article(
            title=str(row.get("title", "")),
            outlet=str(row.get("media_name", "")),
            date=str(row.get("publish_date", "")),
            phase=str(row.get("phase", ""))
        )

        row_dict = row.to_dict()
        row_dict.update(result)
        results.append(row_dict)

        # Save checkpoint every 25 articles
        if (i + 1) % 25 == 0:
            checkpoint = pd.DataFrame(results)
            if resume and os.path.exists(output_csv):
                existing = pd.read_csv(output_csv)
                combined = pd.concat([existing, checkpoint], ignore_index=True)
                combined.to_csv(output_csv, index=False)
            else:
                checkpoint.to_csv(output_csv, index=False)
            print(f"  Checkpoint saved — {i+1} coded")

        time.sleep(delay)

    # Final save
    final = pd.DataFrame(results)
    if resume and os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        final = pd.concat([existing, final], ignore_index=True)

    final.to_csv(output_csv, index=False)

    # Summary
    n_success = final["PE_llm"].notna().sum()
    n_failed  = final["error"].notna().sum() if "error" in final.columns else 0
    print(f"\n=== DONE ===")
    print(f"Total coded: {n_success} / {len(final)}")
    print(f"Failed: {n_failed}")
    print(f"Saved: {output_csv}")

    # Distribution summary
    print(f"\nPE distribution:")
    print(final["PE_llm"].value_counts().sort_index().to_string())
    print(f"\nFraming distribution:")
    print(final["framing_llm"].value_counts().sort_index().to_string())
    print(f"\nConfidence distribution:")
    print(final["confidence_llm"].value_counts().to_string())

    return final


# ── Correlation analysis (after human coding 50 articles) ────────────────────

def correlate(llm_csv, human_csv, output_json="validation_correlations.json"):
    """
    Computes correlations between LLM codes and human codes on the overlap set.

    human_csv must have columns: id, PE_human, framing_human
    (you fill these in manually for 50 articles)
    """
    from scipy.stats import pearsonr, spearmanr

    llm_df   = pd.read_csv(llm_csv)
    human_df = pd.read_csv(human_csv)

    print(f"LLM coded: {len(llm_df)}")
    print(f"Human coded: {len(human_df)}")

    # Merge on id
    merged = llm_df.merge(human_df[["id", "PE_human", "framing_human"]], on="id", how="inner")
    merged = merged.dropna(subset=["PE_llm", "framing_llm", "PE_human", "framing_human"])
    print(f"Overlap (both coded): {len(merged)}")

    if len(merged) < 10:
        print("ERROR: Too few overlapping articles for correlation")
        return

    # PE correlation
    pe_r, pe_p = pearsonr(merged["PE_llm"], merged["PE_human"])
    pe_s, _    = spearmanr(merged["PE_llm"], merged["PE_human"])

    # Framing correlation
    fr_r, fr_p = pearsonr(merged["framing_llm"], merged["framing_human"])
    fr_s, _    = spearmanr(merged["framing_llm"], merged["framing_human"])

    # Mean absolute error
    pe_mae  = float(np.mean(np.abs(merged["PE_llm"] - merged["PE_human"])))
    fr_mae  = float(np.mean(np.abs(merged["framing_llm"] - merged["framing_human"])))

    results = {
        "n_overlap": len(merged),
        "PE": {
            "pearson_r": round(pe_r, 3), "pearson_p": round(pe_p, 4),
            "spearman_rho": round(pe_s, 3), "MAE": round(pe_mae, 3),
            "interpretation": "PASS" if pe_r > 0.7 else "MARGINAL" if pe_r > 0.5 else "FAIL"
        },
        "framing": {
            "pearson_r": round(fr_r, 3), "pearson_p": round(fr_p, 4),
            "spearman_rho": round(fr_s, 3), "MAE": round(fr_mae, 3),
            "interpretation": "PASS" if fr_r > 0.7 else "MARGINAL" if fr_r > 0.5 else "FAIL"
        }
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Correlation Results ===")
    print(f"PE    Pearson r={pe_r:.3f} (p={pe_p:.4f})  Spearman ρ={pe_s:.3f}  MAE={pe_mae:.3f}  [{results['PE']['interpretation']}]")
    print(f"Framing Pearson r={fr_r:.3f} (p={fr_p:.4f})  Spearman ρ={fr_s:.3f}  MAE={fr_mae:.3f}  [{results['framing']['interpretation']}]")
    print(f"\nSaved: {output_json}")

    print(f"""
=== PAPER-READY TEXT ===
For external validation of the LLM coding procedure, we compared LLM-assigned
scores against human judgments on a subset of {len(merged)} articles.
The correlation between LLM and human PE scores was r={pe_r:.2f}
(Spearman ρ={pe_s:.2f}), and between LLM and human framing scores was
r={fr_r:.2f} (Spearman ρ={fr_s:.2f}), indicating
{'acceptable' if pe_r > 0.6 and fr_r > 0.6 else 'partial'} construct validity
for automated coding. LLM coding was then applied to the full 300-article
subset (Gilardi et al., 2023).
""")

    return results


# ── Human coding template (50 articles) ──────────────────────────────────────

def prepare_human_50(llm_csv, output="human_coding_50.csv"):
    """
    Extracts first 50 articles from LLM output for manual validation.
    Creates a simple CSV with empty PE_human and framing_human columns.
    """
    df = pd.read_csv(llm_csv)
    # Take 10 from each phase for balanced human validation
    sample_50 = df.groupby("phase").apply(
        lambda x: x.sample(min(10, len(x)), random_state=42)
    ).reset_index(drop=True)

    out = sample_50[["id", "publish_date", "media_name", "title", "url", "phase",
                     "PE_llm", "framing_llm", "confidence_llm", "reasoning_llm"]].copy()
    out["PE_human"]      = ""
    out["framing_human"] = ""
    out["notes"]         = ""

    out.to_csv(output, index=False)
    print(f"Human coding template saved: {output}")
    print(f"Articles: {len(out)} (10 per phase)")
    print(f"\nInstructions:")
    print(f"  1. Open {output} in Excel")
    print(f"  2. Read each article (click URL)")
    print(f"  3. Fill PE_human (1-5) and framing_human (-2 to +2)")
    print(f"  4. Save and run: python llm_coding.py --step correlate ...")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM coding for SRM external validation")
    parser.add_argument("--step", default="code",
                        choices=["code", "prepare_human", "correlate"],
                        help="Step to run (default: code)")
    parser.add_argument("--input",  default="trump_validation_sample.csv")
    parser.add_argument("--output", default="llm_coded.csv")
    parser.add_argument("--llm",    default="llm_coded.csv",
                        help="LLM coded CSV (for correlate step)")
    parser.add_argument("--human",  default="human_coding_50.csv",
                        help="Human coded CSV (for correlate step)")
    parser.add_argument("--delay",  type=float, default=0.5,
                        help="Delay between API calls in seconds (default: 0.5)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing output")
    args = parser.parse_args()

    if args.step == "code":
        run_coding(args.input, args.output,
                   resume=not args.no_resume, delay=args.delay)

    elif args.step == "prepare_human":
        prepare_human_50(args.llm)

    elif args.step == "correlate":
        correlate(args.llm, args.human)


if __name__ == "__main__":
    main()
