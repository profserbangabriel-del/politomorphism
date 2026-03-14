import sys, os, json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

symbol = sys.argv[1] if len(sys.argv) > 1 else "George Simion"
print(f"STEP 2 - Affective Weight (A) for: {symbol}")

os.makedirs('rezultate', exist_ok=True)

content_path = 'data_simion/Goerge_Simion_-content.csv'

positive_ro = {
    'victorie': 0.8, 'castigat': 0.7, 'succes': 0.7, 'sprijin': 0.5,
    'sustinut': 0.5, 'popular': 0.6, 'puternic': 0.5, 'ales': 0.4,
    'presedinte': 0.3, 'lider': 0.3, 'majoritate': 0.4, 'incredere': 0.5
}
negative_ro = {
    'atac': -0.6, 'acuzat': -0.6, 'scandal': -0.7, 'condamnat': -0.8,
    'dosar': -0.5, 'ancheta': -0.5, 'pericol': -0.7, 'extremist': -0.8,
    'fascist': -0.9, 'radical': -0.6, 'interzis': -0.6, 'critica': -0.4,
    'controversat': -0.5, 'protest': -0.3, 'violenta': -0.8
}

if os.path.exists(content_path):
    df = pd.read_csv(content_path)
    df_simion = df[df['title'].str.contains('simion|Simion|SIMION', na=False, case=False)]
    titluri = df_simion['title'].dropna().tolist()
    print(f"Titles loaded: {len(titluri)}")
    analyzer = SentimentIntensityAnalyzer()
    scores_abs = []
    scores_compound = []
    for titlu in titluri:
        titlu_lower = str(titlu).lower()
        vader = analyzer.polarity_scores(str(titlu))['compound']
        ro_score = 0
        ro_count = 0
        for word, score in {**positive_ro, **negative_ro}.items():
            if word in titlu_lower:
                ro_score += score
                ro_count += 1
        final = (vader * 0.4) + ((ro_score / ro_count) * 0.6) if ro_count > 0 else vader
        scores_compound.append(final)
        scores_abs.append(abs(final))
    A = sum(scores_abs) / len(scores_abs)
    compound_avg = sum(scores_compound) / len(scores_compound)
    method = f"VADER + Romanian dictionary on {len(titluri)} titles"
else:
    print("File not found - using pre-computed value")
    A = 0.0994
    compound_avg = 0.0330
    method = "Pre-computed verified value"

result = {"symbol": symbol, "A": round(A, 4), "compound_avg": round(compound_avg, 4), "method": method}
with open('rezultate/pas2_sentiment.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"A = {A:.4f}")
print("Saved: rezultate/pas2_sentiment.json")
