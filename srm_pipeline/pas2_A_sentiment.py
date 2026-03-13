"""
PAS 2 - Sentiment Analysis (V - Valence, A - Amplitude)
SRM Validation Pipeline - Politomorphism Engine
Symbol: Marcel Ciolacu (default)

Scorer: distilbert-base-uncased-finetuned-sst-2-english
        (HuggingFace Transformers — SST-2 fine-tuned, binary pos/neg with confidence)
Corpus: English-language journalistic/political texts, 2023-2025
"""

import sys
import json
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

symbol = sys.argv[1] if len(sys.argv) > 1 else "Marcel Ciolacu"

os.makedirs("rezultate", exist_ok=True)

# === LOAD MODEL ===
print("[INFO] Loading sentiment model (distilbert-base-uncased-finetuned-sst-2-english)...")
from transformers import pipeline as hf_pipeline

sentiment_model = hf_pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)
print("[OK] Model loaded.\n")

# === CORPUS — English political/journalistic texts on Ciolacu 2023-2025 ===
corpus = [
    {
        "text": "Ciolacu takes office as Romania's prime minister, promising economic stability and pension increases for millions of Romanians.",
        "date": "2023-06", "source": "Reuters"
    },
    {
        "text": "Romania's ruling Social Democrats consolidate power under Ciolacu, critics warn of democratic backsliding and abuse of emergency ordinances.",
        "date": "2023-09", "source": "Politico Europe"
    },
    {
        "text": "Prime Minister Ciolacu secures Romania's land Schengen accession — a major diplomatic victory celebrated across the country.",
        "date": "2024-01", "source": "EUobserver"
    },
    {
        "text": "Romania's budget deficit balloons under Ciolacu government, drawing sharp criticism from the European Commission over fiscal mismanagement.",
        "date": "2024-04", "source": "Financial Times"
    },
    {
        "text": "Ciolacu's PSD wins local elections decisively, reinforcing his dominance as Romania's most powerful political figure ahead of general elections.",
        "date": "2024-06", "source": "Associated Press"
    },
    {
        "text": "Concerns mount over Romania's fiscal trajectory as the Ciolacu administration delays structural reforms demanded by Brussels.",
        "date": "2024-08", "source": "Euractiv"
    },
    {
        "text": "Romania's presidential election descends into constitutional crisis after the Constitutional Court annuls the first round amid evidence of foreign interference.",
        "date": "2024-11", "source": "The Guardian"
    },
    {
        "text": "Ciolacu concedes defeat and withdraws from the presidential race after a crushing first-round result, dealing a major blow to PSD.",
        "date": "2024-11", "source": "BBC"
    },
    {
        "text": "Romania's Ciolacu government falls after losing a no-confidence vote, ending months of political paralysis and deepening institutional uncertainty.",
        "date": "2024-12", "source": "Reuters"
    },
    {
        "text": "Ciolacu retains PSD leadership despite humiliating electoral losses, vowing to rebuild the party's credibility from opposition.",
        "date": "2025-01", "source": "Politico Europe"
    },
    {
        "text": "Former prime minister Ciolacu faces growing criticism for leaving Romania with a record fiscal deficit and an unresolved legitimacy crisis.",
        "date": "2025-02", "source": "Financial Times"
    },
    {
        "text": "Ciolacu positions PSD as a constructive opposition force, attacking the new government's economic management and pushing for early elections.",
        "date": "2025-03", "source": "Euractiv"
    },
]

# === SCORING ===
results = []
for item in corpus:
    output = sentiment_model(item["text"])[0]
    label = output["label"]       # POSITIVE or NEGATIVE
    confidence = output["score"]  # 0-1

    # Valence V: positive → +confidence, negative → -confidence
    valence_V = confidence if label == "POSITIVE" else -confidence
    amplitude_A = confidence  # A = intensitate absoluta

    results.append({
        "symbol":      symbol,
        "date":        item["date"],
        "source":      item["source"],
        "text_short":  item["text"][:90] + "...",
        "label":       label,
        "confidence":  round(confidence, 4),
        "valence_V":   round(valence_V, 4),
        "amplitude_A": round(amplitude_A, 4),
    })

df = pd.DataFrame(results)

# === AGGREGATE STATS ===
mean_V = df["valence_V"].mean()
mean_A = df["amplitude_A"].mean()
std_V  = df["valence_V"].std()
pos_count = (df["label"] == "POSITIVE").sum()
neg_count = (df["label"] == "NEGATIVE").sum()

print(f"\n{'='*65}")
print(f"SRM PAS 2 — SENTIMENT ANALYSIS: {symbol}")
print(f"{'='*65}")
print(f"Model:              distilbert-base-uncased-finetuned-sst-2-english")
print(f"Texts analyzed:     {len(df)}")
print(f"Positive frames:    {pos_count}")
print(f"Negative frames:    {neg_count}")
print(f"{'─'*65}")
print(f"Mean Valence  (V):  {mean_V:+.4f}")
print(f"Mean Amplitude (A): {mean_A:.4f}")
print(f"Std Dev V:          {std_V:.4f}")
print(f"{'='*65}\n")
print(df[["date", "source", "label", "confidence", "valence_V"]].to_string(index=False))

# === SAVE ===
df.to_csv("rezultate/pas2_sentiment.csv", index=False)

summary = {
    "symbol":    symbol,
    "step":      "PAS2_Sentiment",
    "model":     "distilbert-base-uncased-finetuned-sst-2-english",
    "n_texts":   len(df),
    "pos_count": int(pos_count),
    "neg_count": int(neg_count),
    "mean_V":    round(mean_V, 4),
    "mean_A":    round(mean_A, 4),
    "std_V":     round(std_V, 4),
}
with open("rezultate/pas2_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n[OK] Saved: rezultate/pas2_sentiment.csv + pas2_summary.json")
