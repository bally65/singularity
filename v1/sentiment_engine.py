import os
import requests
import json
import time

def fetch_crypto_sentiment():
    """
    Fetches real-time crypto sentiment indicators from Fear & Greed Index or news snippets.
    Returns a score from -1 (Extremely Negative) to 1 (Extremely Positive).
    """
    print("📰 Fetching Macro Market Sentiment...")
    try:
        # 1. Fear & Greed Index (A reliable macro proxy)
        response = requests.get("https://api.alternative.me/fng/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            value = int(data['data'][0]['value'])
            # Convert 0-100 to -1 to 1
            sentiment = (value - 50) / 50.0
            print(f"   Fear & Greed Index: {value} (Score: {sentiment:.2f})")
            return sentiment
    except Exception as e:
        print(f"   ⚠️ Sentiment fetch failed: {e}")
    
    return 0.0 # Neutral fallback

def save_sentiment():
    score = fetch_crypto_sentiment()
    path = "/home/aa598/.openclaw/workspace/singularity/project/v1/macro_sentiment.json"
    with open(path, 'w') as f:
        json.dump({"score": score, "timestamp": time.time()}, f)
    print(f"✅ Macro sentiment saved to {path}")

if __name__ == "__main__":
    save_sentiment()
