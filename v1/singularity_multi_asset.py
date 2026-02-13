import onnxruntime as ort
import pandas as pd
import numpy as np
import time
import os
import json
from io import StringIO
from collections import deque

# Configuration
DATA_DIR = '/home/aa598/.openclaw/workspace/singularity/project/v2'
MODEL_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v1/singularity_v3.onnx'
LOG_FILE = '/home/aa598/.openclaw/workspace/singularity/project/v1/challenge_multi_asset.log'

class MultiAssetTrader:
    def __init__(self, initial_usdt=281.25):
        self.usdt = initial_usdt
        self.portfolios = {
            'BTCUSDT': {'pos': 0.0, 'entry': 0.0, 'equity': initial_usdt / 3},
            'ETHUSDT': {'pos': 0.0, 'entry': 0.0, 'equity': initial_usdt / 3},
            'SOLUSDT': {'pos': 0.0, 'entry': 0.0, 'equity': initial_usdt / 3}
        }
        self.total_usdt = initial_usdt
        self.sess = ort.InferenceSession(MODEL_PATH)
        self.input_name = self.sess.get_inputs()[0].name
        
        # Symbol histories for smoothing
        self.histories = {s: deque(maxlen=20) for s in self.portfolios.keys()}
        
        self.log(f"🌌 Singularity Multi-Asset Challenge Started (Total: {initial_usdt} USDT)")

    def log(self, msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}\n"
        print(line, end='', flush=True)
        with open(LOG_FILE, 'a') as f:
            f.write(line)

    def get_features(self, df_tail):
        df = df_tail.copy()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # --- Optimization: Advanced Technical Indicators ---
        # 1. Momentum & Volatility
        df['price_change'] = df['price'].diff().fillna(0)
        df['velocity'] = df['price_change']
        df['accel'] = df['velocity'].diff().fillna(0)
        df['std'] = df['price'].rolling(window=20).std().fillna(0)
        
        # 2. RSI (Relative Strength Index) - 14 period
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50) / 100.0 # Normalized 0-1
        
        # 3. Bollinger Bands Distance
        df['ma20'] = df['price'].rolling(window=20).mean().fillna(df['price'])
        df['upper'] = df['ma20'] + (df['std'] * 2)
        df['lower'] = df['ma20'] - (df['std'] * 2)
        # Distance to bands: +1 at upper, -1 at lower
        df['bb_pos'] = (df['price'] - df['ma20']) / (df['std'] * 2 + 1e-6)
        
        # 4. Entropy & Mass
        df['entropy'] = df['price_change'].rolling(window=50).std().fillna(0)
        df['mass'] = df['qty'].rolling(window=20).mean().fillna(0)
        df['imbalance'] = np.where(df['maker'] == True, -1, 1) * df['qty']
        
        # We now have more features. Let's select the most impactful ones
        # and ensure the model input dimension matches (v3 uses 6 features)
        # If we want to change input dim, we'd need to re-train. 
        # For now, let's keep it to 6 but use better indicators.
        
        # Selected: velocity, accel, entropy, rsi, bb_pos, imbalance
        feature_cols = ['velocity', 'accel', 'entropy', 'rsi', 'bb_pos', 'imbalance']
        
        features = df[feature_cols].tail(60).values
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
        return features.astype(np.float32).reshape(1, 60, 6)

    def run_cycle(self):
        current_total_equity = 0
        
        # Load Macro Sentiment
        macro_sentiment = 0.0
        sentiment_path = os.path.join(os.path.dirname(MODEL_PATH), "macro_sentiment.json")
        if os.path.exists(sentiment_path):
            with open(sentiment_path, 'r') as f:
                sentiment_data = json.load(f)
                # Expire after 1 hour
                if time.time() - sentiment_data['timestamp'] < 3600:
                    macro_sentiment = sentiment_data['score']

        for symbol in self.portfolios.keys():
            try:
                # ... existing data loading ...
                data_path = os.path.join(DATA_DIR, f"dataset_{symbol.lower()}.csv")
                if not os.path.exists(data_path): continue
                
                raw_tail = os.popen(f'tail -n 1000 {data_path}').read()
                df = pd.read_csv(StringIO(raw_tail), names=['ts_now', 'ts_event', 'price', 'qty', 'maker'], on_bad_lines='skip', engine='python').apply(pd.to_numeric, errors='coerce').dropna()
                
                if len(df) < 150: continue
                current_price = df['price'].iloc[-1]
                if current_price <= 0: return # Stop whole loop if price sync fails

                volatility = df['price'].tail(100).std()
                features = self.get_features(df.tail(150))
                
                raw_pred = self.sess.run(None, {self.input_name: features})[0]
                self.histories[symbol].append(raw_pred[0][1])
                smooth_pred = np.mean(self.histories[symbol])
                
                # --- Adaptive Strategy with Sentiment Filter ---
                base_threshold = 0.012 # Slightly lowered from 0.015 to catch bounces
                # If sentiment is negative, increase entry threshold
                sentiment_adj = 1.0 - (macro_sentiment * 0.4) # Slightly less aggressive filter
                dynamic_threshold = base_threshold * (1.0 + (volatility / (current_price / 1000.0))) * sentiment_adj
                
                # If sentiment is extremely negative (<-0.7), block all entries
                if macro_sentiment < -0.7:
                    dynamic_threshold = 999.0 
                dynamic_stop_pct = min(0.01, 0.003 * (1.0 + (volatility / (current_price / 5000.0))))
                
                p = self.portfolios[symbol]
                
                # Allocation: Use a third of total USDT for each if empty
                alloc_usdt = self.usdt / len(self.portfolios)
                
                if smooth_pred > dynamic_threshold and p['pos'] == 0:
                    p['pos'] = alloc_usdt * 0.9 / current_price
                    p['entry'] = current_price
                    self.usdt -= (alloc_usdt * 0.9)
                    self.log(f"🚀 [ENTRY] {symbol} @ {current_price:.2f} | Pred: {smooth_pred:.4f} | Stop: {dynamic_stop_pct*100:.2f}%")
                
                elif smooth_pred < -dynamic_threshold * 0.5 and p['pos'] > 0:
                    sale_usdt = p['pos'] * current_price
                    self.usdt += sale_usdt
                    self.log(f"📉 [EXIT] {symbol} @ {current_price:.2f} | Pred: {smooth_pred:.4f}")
                    p['pos'] = 0
                    p['equity'] = sale_usdt

                if p['pos'] > 0:
                    p['equity'] = p['pos'] * current_price
                    if current_price < p['entry'] * (1.0 - dynamic_stop_pct):
                        sale_usdt = p['pos'] * current_price
                        self.usdt += sale_usdt
                        self.log(f"🛑 [STOP] {symbol} @ {current_price:.2f} | Loss: -{dynamic_stop_pct*100:.2f}%")
                        p['pos'] = 0
                        p['equity'] = sale_usdt
                
                current_total_equity += p['equity'] if p['pos'] > 0 else (alloc_usdt)
                
            except Exception:
                continue

        if time.time() % 300 < 1:
            self.log(f"📊 Heartbeat | Total Equity approx: {self.usdt + sum(p['equity'] for p in self.portfolios.values() if p['pos'] > 0):.2f} USDT")

if __name__ == "__main__":
    trader = MultiAssetTrader()
    print("🌌 Singularity Multi-Asset Trader Active (BTC, ETH, SOL)...", flush=True)
    while True:
        trader.run_cycle()
        time.sleep(2)
