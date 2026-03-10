import sys, os, json, re
import numpy as np
from collections import defaultdict

os.makedirs("rezultate", exist_ok=True)

SIMBOL = sys.argv[1] if len(sys.argv) > 1 else "sunflower movement"

CORPUS_T0 = [
    "taiwan cross strait relations china economic cooperation framework agreement trade",
    "kuomintang party taiwan politics legislature yuan democratic progressive opposition",
    "taiwan parliament legislature national assembly political debate trade policy",
    "cross strait economic integration trade agreement service sector bilateral talks",
    "ma ying jeou government taiwan china policy economic integration diplomacy",
]

CORPUS_T1 = [
    "sunflower movement students occupy taiwan legislature parliament protest democracy",
    "sunflower revolution youth democracy taiwan cross strait service trade agreement opposition",
    "students storm parliament building taiwan protest occupy legislature civil disobedience",
    "sunflower movement taiwan democracy protest civil society mobilization resistance",
    "occupy legislature taiwan students blockade parliament democracy rights sovereignty",
    "sunflower protest taiwan youth occupy government building democratic rights",
    "hundreds thousands rally taiwan sunflower movement support student protest solidarity",
    "sunflower occupation taiwan parliament youth democracy movement success victory",
]

def tokenize(text):
    return re.findall(r'\b[a-z]{3,}\b', text.lower())

def build_context(corpus, targets, window=5):
    ctx = defaultdict(float)
    for doc in corpus:
        tokens = tokenize(doc)
        for i, tok in enumerate(tokens):
            if tok in targets:
                for j in range(max(0,i-window), min(len(tokens),i+window+1)):
                    if j != i and tokens[j] not in targets:
                        ctx[tokens[j]] += 1.0
    return ctx

def cosine(v1, v2):
    common = set(v1) & set(v2)
    if not common: return 0.0
    num = sum(v1[k]*v2[k] for k in common)
    d1 = np.sqrt(sum(x**2 for x in v1.values()))
    d2 = np.sqrt(sum(x**2 for x in v2.values()))
    return num / (d1 * d2) if d1 and d2 else 0.0

TARGET = {"sunflower","movement","taiwan","protest","democracy","parliament"}
ctx0 = build_context(CORPUS_T0, TARGET)
ctx1 = build_context(CORPUS_T1, TARGET)
D = 1.0 - cosine(ctx0, ctx1)

print(f"[PAS 3] D = {D:.4f}")

with open("rezultate/rezultat_D.json", "w") as f:
    json.dump({"D": round(D, 4)}, f)
