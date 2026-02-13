import onnxruntime as ort
import pandas as pd
import numpy as np
import os
import json
from io import StringIO

# Configuration
DATA_DIR = '/home/aa598/.openclaw/workspace/singularity/project/v2'
MODEL_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v1/singularity_v3.onnx'
SENTIMENT_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v1/macro_sentiment.json'

def process_df(df_tail):
    df = df_tail.copy()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['price_change'] = df['price'].diff().fillna(0)
    df['velocity'] = df['price_change']
    df['accel'] = df['velocity'].diff().fillna(0)
    df['std'] = df['price'].rolling(window=20).std().fillna(0)
    
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi_raw'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi_raw'].fillna(50) / 100.0
    
    df['ma20'] = df['price'].rolling(window=20).mean().fillna(df['price'])
    df['bb_pos'] = (df['price'] - df['ma20']) / (df['std'] * 2 + 1e-6)
    df['entropy'] = df['price_change'].rolling(window=50).std().fillna(0)
    df['imbalance'] = np.where(df['maker'] == True, -1, 1) * df['qty']
    return df

def analyze_assets():
    sess = ort.InferenceSession(MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    
    # Load Sentiment
    sentiment = 0.0
    if os.path.exists(SENTIMENT_PATH):
        with open(SENTIMENT_PATH, 'r') as f:
            sentiment = json.load(f).get('score', 0.0)
    
    results = {}
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        path = os.path.join(DATA_DIR, f"dataset_{symbol.lower()}.csv")
        if not os.path.exists(path): continue
        
        raw_tail = os.popen(f'tail -n 1000 {path}').read()
        df = pd.read_csv(StringIO(raw_tail), names=['ts_now', 'ts_event', 'price', 'qty', 'maker'], on_bad_lines='skip', engine='python').apply(pd.to_numeric, errors='coerce').dropna()
        
        df = process_df(df)
        feature_cols = ['velocity', 'accel', 'entropy', 'rsi', 'bb_pos', 'imbalance']
        feat_matrix = df[feature_cols].tail(60).values
        feat_matrix = (feat_matrix - np.mean(feat_matrix, axis=0)) / (np.std(feat_matrix, axis=0) + 1e-6)
        seq = feat_matrix.astype(np.float32).reshape(1, 60, 6)
        
        pred = sess.run(None, {input_name: seq})[0][0][1]
        
        results[symbol] = {
            "price": df['price'].iloc[-1],
            "prediction": pred,
            "rsi": df['rsi_raw'].iloc[-1]
        }
    
    print(f"--- 🌌 Singularity Intel (Post-Optimization) ---")
    print(f"Macro Sentiment Score: {sentiment:.2f}")
    for s, r in results.items():
        bias = "BULLISH" if r['prediction'] > 0.015 else ("BEARISH" if r['prediction'] < -0.01 else "NEUTRAL")
        print(f"🔹 {s}: ${r['price']:.2f} | Pred: {r['prediction']:.4f} | RSI: {r['rsi']:.1f} | Bias: {bias}")

if __name__ == "__main__":
    analyze_assets()
