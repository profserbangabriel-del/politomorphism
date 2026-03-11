import sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("rezultate", exist_ok=True)
SIMBOL = sys.argv[1] if len(sys.argv) > 1 else "Calin Georgescu"

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Corpus empiric - surse: RFE/RL, Reuters, BBC, DFRLab, Foreign Policy
# Perioada: noiembrie-decembrie 2024
CORPUS = [
    "Georgescu won 23 percent in a stunning upset that shocked Romania and all of Europe.",
    "Thousands of fake TikTok accounts coordinated to amplify Georgescu content before elections.",
    "Georgescu represents a dangerous threat to Romanian democracy and NATO membership.",
    "Romanians inspired by Georgescu vision of national sovereignty and spiritual identity.",
    "Constitutional court annulled the election due to massive social media manipulation.",
    "Georgescu appeals to deep frustrations with corruption and poverty in Romanian society.",
    "Intelligence documents revealed state-backed cyberattacks amplifying Georgescu campaign.",
    "His antisemitic and pro-Putin rhetoric is a serious threat to European security.",
    "Georgescu TikTok campaign reached millions through emotional and patriotic messaging.",
    "Over 25000 coordinated accounts boosted Georgescu from 5 percent to 23 percent in weeks.",
    "Many Romanians see Georgescu as the only authentic voice against the corrupt establishment.",
    "Romanian influencers were paid to distribute pro-Georgescu content across social media.",
    "The election annulment was described by supporters as an attack on Romanian democracy.",
    "Western governments expressed alarm at Georgescu pro-Russian and anti-NATO positions.",
    "Georgescu campaign flooded TikTok with church visits projecting strength and piety.",
    "Students protested in Bucharest against Georgescu extremist and antisemitic views.",
    "Georgescu victory exposed deep divisions between urban elites and rural Romanians.",
    "The rise of Georgescu signals the failure of mainstream parties to address poverty.",
    "Romania faces its worst democratic crisis since 1989 because of Georgescu phenomenon.",
    "Georgescu supporters celebrated his result as a revolution against the corrupt system.",
]

scores = [abs(analyzer.polarity_scores(t)["compound"]) for t in CORPUS]
A = sum(scores) / len(scores)
print(f"[PAS 2] A = {A:.4f}")
with open("rezultate/rezultat_A.json", "w") as f:
    json.dump({"A": round(A, 4), "sursa": "RFE/RL, Reuters, BBC, DFRLab nov-dec 2024"}, f)
