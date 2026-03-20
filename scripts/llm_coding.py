"""
llm_coding.py — v2
Politomorphism Research Project | Serban Gabriel Florin
License: CC BY 4.0

Automated coding of PE and Framing using Claude API.
Fixed: uses requests library, reads ANTHROPIC_API_KEY from environment.
"""

import pandas as pd
import numpy as np
import json
import time
import argparse
import os
import re
import sys

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a political communication researcher coding news articles.
You will receive a news article title and outlet.
Return ONLY valid JSON, no explanation, no markdown.

SCORE 1 — PE (Topical Diversity), scale 1-5:
1 = Single topic only
2 = One main topic + one secondary
3 = Two distinct topics balanced
4 = Three or more topics
5 = Highly diverse, unrelated topics

Topics: electoral campaign, immigration, economy, foreign policy,
media/Twitter, Republican Party, legal/judicial, race/religion,
Clinton/Democrats, personal/biography

SCORE 2 — FRAMING toward Trump, scale -2 to +2:
-2 = Strongly anti-Trump
-1 = Mildly anti-Trump
 0 = Neutral/descriptive
+1 = Mildly pro-Trump
+2 = Strongly pro-Trump

Return ONLY this JSON:
{"PE": <1-5>, "framing": <-2 to 2>, "confidence": "high"|"medium"|"low", "reasoning": "<one sentence>"}"""


# ── Single article coding ─────────────────────────────────────────────────────

def code_article(title, outlet, date, phase, api_key, max_retries=3):
    try:
        import requests
    except ImportError:
        os.system(f"{sys.executable} -m pip install requests -q")
        import requests

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 200,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": f"Title: {title}\nOutlet: {outlet}\nDate: {date}\nPhase: {phase}"
        }]
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()
                text = data["content"][0]["text"].strip()
                text = re.sub(r"```json\s*", "", text)
                text = re.sub(r"```\s*", "", text)
                result = json.loads(text)

                return {
                    "PE_llm":         int(result["PE"]),
                    "framing_llm":    int(result["framing"]),
                    "confidence_llm": result.get("confidence", "medium"),
                    "reasoning_llm":  result.get("reasoning", ""),
                    "error":          None
                }

            elif resp.status_code == 429:
                wait = (attempt + 1) * 30
                print(f"    Rate limit — waiting {wait}s...")
                time.sleep(wait)

            elif resp.status_code == 529:
                wait = (attempt + 1) * 15
                print(f"    Overloaded — waiting {wait}s...")
                time.sleep(wait)

            else:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
                break

        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)

        except Exception as e:
            print(f"    Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                break

    return {
        "PE_llm": None, "framing_llm": None,
        "confidence_llm": None, "reasoning_llm": None,
        "error": "FAILED"
    }


# ── Main coding loop ──────────────────────────────────────────────────────────

def run_coding(input_csv, output_csv, delay=0.3):

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    print(f"API key loaded: {api_key[:12]}...")

    df = pd.read_csv(input_csv)
    df = df.drop_duplicates(subset=["id"]).head(300)
    print(f"Articles to code: {len(df)}")

    coded_ids = set()
    existing_rows = []
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        existing = existing.drop_duplicates(subset=["id"])
        done = existing[existing["PE_llm"].notna()]
        coded_ids = set(done["id"].tolist())
        existing_rows = done.to_dict("records")
        print(f"Already coded: {len(coded_ids)}")

    to_code = df[~df["id"].isin(coded_ids)]
    print(f"Remaining: {len(to_code)}")

    results = list(existing_rows)

    for i, (_, row) in enumerate(to_code.iterrows()):
        title  = str(row.get("title", ""))[:200]
        outlet = str(row.get("media_name", ""))
        date   = str(row.get("publish_date", ""))
        phase  = str(row.get("phase", ""))

        print(f"[{i+1}/{len(to_code)}] {outlet} | {title[:55]}...")

        result = code_article(title, outlet, date, phase, api_key)
        row_dict = row.to_dict()
        row_dict.update(result)
        results.append(row_dict)

        if (i + 1) % 25 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)
            n_ok = sum(1 for r in results if r.get("PE_llm") is not None)
            print(f"  Checkpoint: {n_ok}/{len(results)} ok")

        time.sleep(delay)

    final = pd.DataFrame(results)
    final.to_csv(output_csv, index=False)

    n_ok   = final["PE_llm"].notna().sum()
    n_fail = (final["error"] == "FAILED").sum()

    print(f"\n=== DONE ===")
    print(f"Total: {len(final)} | Success: {n_ok} | Failed: {n_fail}")
    print(f"Saved: {output_csv}")

    if n_ok > 0:
        print(f"\nPE distribution:")
        print(final["PE_llm"].value_counts().sort_index().to_string())
        print(f"\nFraming distribution:")
        print(final["framing_llm"].value_counts().sort_index().to_string())


# ── Correlate ─────────────────────────────────────────────────────────────────

def correlate(llm_csv, human_csv):
    from scipy.stats import pearsonr, spearmanr

    llm   = pd.read_csv(llm_csv)
    human = pd.read_csv(human_csv)

    merged = llm.merge(
        human[["id", "PE_human", "framing_human"]],
        on="id", how="inner"
    ).dropna(subset=["PE_llm", "framing_llm", "PE_human", "framing_human"])

    print(f"Overlap: {len(merged)} articles")

    pe_r, pe_p = pearsonr(merged["PE_llm"], merged["PE_human"])
    fr_r, fr_p = pearsonr(merged["framing_llm"], merged["framing_human"])
    pe_s, _    = spearmanr(merged["PE_llm"], merged["PE_human"])
    fr_s, _    = spearmanr(merged["framing_llm"], merged["framing_human"])

    print(f"\nPE:      r={pe_r:.3f} (p={pe_p:.4f})  rho={pe_s:.3f}")
    print(f"Framing: r={fr_r:.3f} (p={fr_p:.4f})  rho={fr_s:.3f}")

    interp_pe = "PASS" if pe_r > 0.7 else "MARGINAL" if pe_r > 0.5 else "FAIL"
    interp_fr = "PASS" if fr_r > 0.7 else "MARGINAL" if fr_r > 0.5 else "FAIL"

    print(f"\nPE validation: {interp_pe}")
    print(f"Framing validation: {interp_fr}")

    print(f"""
=== PAPER TEXT ===
LLM coding validated against human judgments on {len(merged)} articles.
PE: r={pe_r:.2f} (rho={pe_s:.2f}). Framing: r={fr_r:.2f} (rho={fr_s:.2f}).
(Gilardi et al., 2023)
""")


# ── Prepare human 50 ─────────────────────────────────────────────────────────

def prepare_human_50(llm_csv):
    df = pd.read_csv(llm_csv)
    df = df[df["PE_llm"].notna()]

    sample = df.groupby("phase").apply(
        lambda x: x.sample(min(10, len(x)), random_state=42)
    ).reset_index(drop=True)

    out = sample[["id", "publish_date", "media_name", "title", "url",
                  "phase", "PE_llm", "framing_llm", "reasoning_llm"]].copy()
    out["PE_human"]      = ""
    out["framing_human"] = ""
    out["notes"]         = ""

    out.to_csv("human_coding_50.csv", index=False)
    print(f"Saved: human_coding_50.csv ({len(out)} articles)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",   default="code",
                        choices=["code", "prepare_human", "correlate"])
    parser.add_argument("--input",  default="trump_validation_sample.csv")
    parser.add_argument("--output", default="llm_coded.csv")
    parser.add_argument("--llm",    default="llm_coded.csv")
    parser.add_argument("--human",  default="human_coding_50.csv")
    parser.add_argument("--delay",  type=float, default=0.3)
    args = parser.parse_args()

    if args.step == "code":
        run_coding(args.input, args.output, delay=args.delay)
    elif args.step == "prepare_human":
        prepare_human_50(args.llm)
    elif args.step == "correlate":
        correlate(args.llm, args.human)


if __name__ == "__main__":
    main()
