import pandas as pd
import numpy as np
import os
import json
import time
from io import StringIO
import onnxruntime as ort

# Configuration
DATA_DIR = '/home/aa598/.openclaw/workspace/singularity/project/v2'
MODEL_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v1/singularity_v3.onnx'
SENTIMENT_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v1/macro_sentiment.json'

def get_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

def check_bottom_fishing():
    print("🎣 Scanning for Bottom Fishing Opportunities...")
    
    # Load Sentiment
    sentiment_score = 0
    if os.path.exists(SENTIMENT_PATH):
        with open(SENTIMENT_PATH, 'r') as f:
            sentiment_score = json.load(f).get('score', 0)

    # Initialize model
    sess = ort.InferenceSession(MODEL_PATH)
    input_name = sess.get_inputs()[0].name

    results = []
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        path = os.path.join(DATA_DIR, f"dataset_{symbol.lower()}.csv")
        if not os.path.exists(path): continue
        
        raw_tail = os.popen(f'tail -n 1000 {path}').read()
        df = pd.read_csv(StringIO(raw_tail), names=['ts_now', 'ts_event', 'price', 'qty', 'maker'], on_bad_lines='skip', engine='python').apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df) < 100: continue
        
        # Calculate RSI
        df['rsi'] = get_rsi(df['price'])
        current_rsi = df['rsi'].iloc[-1]
        current_price = df['price'].iloc[-1]
        
        # Prediction
        # (Simplified feature prep for alert)
        df['pc'] = df['price'].diff().fillna(0)
        df['vel'] = df['pc']
        df['acc'] = df['vel'].diff().fillna(0)
        df['ent'] = df['pc'].rolling(window=50).std().fillna(0)
        df['ma20'] = df['price'].rolling(window=20).mean().fillna(df['price'])
        df['std'] = df['price'].rolling(window=20).std().fillna(0)
        df['bb'] = (df['price'] - df['ma20']) / (df['std'] * 2 + 1e-6)
        df['imb'] = np.where(df['maker'] == True, -1, 1) * df['qty']
        
        feats = df[['vel', 'acc', 'ent', 'rsi', 'bb', 'imb']].tail(60).values
        feats = (feats - np.mean(feats, axis=0)) / (np.std(feats, axis=0) + 1e-6)
        pred = sess.run(None, {input_name: feats.astype(np.float32).reshape(1, 60, 6)})[0][0][1]

        # 抄底條件：
        # 1. RSI < 30 (超賣)
        # 2. V3 預測值轉正 (出現底部分歧)
        # 3. 市場處於極度恐懼 (Sentiment < -0.5)
        is_bottom = (current_rsi < 35) and (pred > 0.005) and (sentiment_score < -0.4)
        
        if is_bottom:
            results.append({
                "symbol": symbol,
                "price": current_price,
                "rsi": current_rsi,
                "pred": pred,
                "strength": "🔥 STRONG BUY" if current_rsi < 20 else "✅ BUY"
            })

    if not results:
        # print("   - No strong bottom signals yet. Market still cooling.")
        pass # Silent if no results
    else:
        print("🚨 BOTTOM FISHING ALERT! 🚨")
        for r in results:
            print(f"   [{r['strength']}] {r['symbol']}: \${r['price']} (RSI: {r['rsi']:.1f}, V3: {r['pred']:.4f})")

if __name__ == "__main__":
    check_bottom_fishing()
