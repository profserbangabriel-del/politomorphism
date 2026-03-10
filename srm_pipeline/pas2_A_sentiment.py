import sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("rezultate", exist_ok=True)

SIMBOL = sys.argv[1] if len(sys.argv) > 1 else "sunflower movement"

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

CORPUS = [
    "The sunflower movement was a remarkable display of youth democratic activism in Taiwan.",
    "Students occupied the legislature to defend Taiwan sovereignty and democratic rights.",
    "The movement succeeded in halting the controversial trade agreement with mainland China.",
    "Hundreds of thousands rallied in support of the sunflower student protesters in Taipei.",
    "The peaceful occupation demonstrated the strength of Taiwanese civil society.",
    "International observers praised the discipline and organization of Taiwan protesters.",
    "The sunflower movement energized a generation of Taiwanese youth.",
    "Protesters showed incredible courage facing potential government crackdown.",
    "The illegal occupation of the legislature undermined Taiwan rule of law.",
    "Student protesters blocked legitimate democratic processes with unlawful sit-in.",
    "The cross-strait service agreement would have benefited Taiwan economy significantly.",
    "Protesters were manipulated by opposition parties for political gain.",
    "The occupation set a dangerous precedent for mob rule over democratic institutions.",
    "The sunflower occupation marked a turning point in Taiwan democratic consciousness.",
    "An estimated 500,000 people participated in the largest rally on March 30.",
]

scores = [abs(analyzer.polarity_scores(t)["compound"]) for t in CORPUS]
A = sum(scores) / len(scores)

print(f"[PAS 2] A = {A:.4f}")

with open("rezultate/rezultat_A.json", "w") as f:
    json.dump({"A": round(A, 4)}, f)
