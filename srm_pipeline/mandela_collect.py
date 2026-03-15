import requests
import json
import time
import csv
import os

API_KEY = os.environ.get('NYT_API_KEY', '')

def get_archive_month(year, month, api_key):
    """Get all articles for a month from NYT Archive API."""
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            docs = r.json().get('response', {}).get('docs', [])
            # Filter for mandela mentions
            mandela_docs = []
            for d in docs:
                title = d.get('headline', {}).get('main', '')
                abstract = d.get('abstract', '') or ''
                snippet = d.get('snippet', '') or ''
                text = f"{title} {abstract} {snippet}".lower()
                if 'mandela' in text:
                    mandela_docs.append({
                        'publish_date': d.get('pub_date', '')[:10],
                        'title': title,
                        'media_name': 'nytimes.com'
                    })
            print(f"  {year}-{month:02d}: {len(docs)} total articles, {len(mandela_docs)} with mandela")
            return len(mandela_docs), mandela_docs
        elif r.status_code == 429:
            print("  Rate limit - waiting 30s...")
            time.sleep(30)
            return get_archive_month(year, month, api_key)
        else:
            print(f"  Error {r.status_code}")
            return 0, []
    except Exception as e:
        print(f"  Error: {e}")
        return 0, []

print("=== NYT Archive API - Mandela Data Collection ===")
print(f"API Key: {'SET' if API_KEY else 'NOT SET'}")

if not API_KEY:
    print("ERROR: NYT_API_KEY not set")
    exit(1)

os.makedirs('data_mandela', exist_ok=True)

# BASELINE: 1988-01 to 1990-01
print("\nCollecting BASELINE (1988-01 to 1990-01)...")
baseline_rows = []
for year in [1988, 1989]:
    for month in range(1, 13):
        count, _ = get_archive_month(year, month, API_KEY)
        baseline_rows.append({'date': f"{year}-{month:02d}-01", 'count': count, 'year': year, 'month': month})
        time.sleep(12)  # Archive API: 5 req/min

# Jan 1990
count, _ = get_archive_month(1990, 1, API_KEY)
baseline_rows.append({'date': '1990-01-01', 'count': count, 'year': 1990, 'month': 1})
time.sleep(12)

with open('data_mandela/mandela_baseline.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['date', 'count', 'year', 'month'])
    writer.writeheader()
    writer.writerows(baseline_rows)
print(f"Baseline saved: {len(baseline_rows)} months")

# ANALYSIS: 1990-02 to 1991-12
print("\nCollecting ANALYSIS (1990-02 to 1991-12)...")
analysis_rows = []
all_titles = []

for year in [1990, 1991]:
    start_month = 2 if year == 1990 else 1
    for month in range(start_month, 13):
        count, titles = get_archive_month(year, month, API_KEY)
        analysis_rows.append({'date': f"{year}-{month:02d}-01", 'count': count, 'year': year, 'month': month})
        all_titles.extend(titles)
        time.sleep(12)

with open('data_mandela/mandela_analysis.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['date', 'count', 'year', 'month'])
    writer.writeheader()
    writer.writerows(analysis_rows)

with open('data_mandela/mandela_titles.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['publish_date', 'title', 'media_name'])
    writer.writeheader()
    writer.writerows(all_titles)

print(f"Analysis saved: {len(analysis_rows)} months")
print(f"Titles saved: {len(all_titles)}")

# Summary
total_baseline = sum(r['count'] for r in baseline_rows)
total_analysis = sum(r['count'] for r in analysis_rows)
print(f"\nTotal baseline articles: {total_baseline}")
print(f"Total analysis articles: {total_analysis}")
print(f"Peak month: {max(analysis_rows, key=lambda x: x['count'])}")
print("Done!")
