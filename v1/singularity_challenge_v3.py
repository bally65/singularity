import onnxruntime as ort
import pandas as pd
import numpy as np
import time
import os
from io import StringIO
from collections import deque

# Configuration
V2_DATA_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v2/dataset_v2.csv'
MODEL_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v1/singularity_v3.onnx'
PORTFOLIO_FILE = '/home/aa598/.openclaw/workspace/singularity/project/v1/challenge_9000.log'

class OptimizedChallengeTraderV3:
    def __init__(self, initial_usdt=281.25):
        self.usdt = initial_usdt
        self.position = 0.0 # BTC
        self.entry_price = 0.0
        self.equity = initial_usdt
        self.sess = ort.InferenceSession(MODEL_PATH)
        self.input_name = self.sess.get_inputs()[0].name
        
        # Performance Smoothing
        self.pred_history = deque(maxlen=20) # Increased for stability
        
        if not os.path.exists(PORTFOLIO_FILE):
            self.log("🌌 Singularity V3 Adaptive Optimization Started")

    def log(self, msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}\n"
        print(line, end='', flush=True)
        with open(PORTFOLIO_FILE, 'a') as f:
            f.write(line)

    def get_features(self, df_tail):
        df = df_tail.copy()
        df['price_change'] = df['price'].diff().fillna(0)
        df['velocity'] = df['price_change']
        df['accel'] = df['velocity'].diff().fillna(0)
        df['entropy'] = df['price_change'].rolling(window=50).std().fillna(0)
        df['mass'] = df['qty'].rolling(window=20).mean().fillna(0)
        df['imbalance'] = np.where(df['maker'] == True, -1, 1) * df['qty']
        df['liq_force'] = 0 
        
        features = df[['velocity', 'accel', 'entropy', 'mass', 'imbalance', 'liq_force']].tail(60).values
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
        return features.astype(np.float32).reshape(1, 60, 6)

    def run_cycle(self):
        try:
            raw_tail = os.popen(f'tail -n 1500 {V2_DATA_PATH}').read()
            df = pd.read_csv(StringIO(raw_tail), names=['ts_now', 'ts_event', 'price', 'qty', 'maker'], on_bad_lines='skip', engine='python').apply(pd.to_numeric, errors='coerce').dropna()
            
            if len(df) < 150: return
            current_price = df['price'].iloc[-1]
            if current_price <= 0: return

            volatility = df['price'].tail(100).std()
            features = self.get_features(df.tail(150))
            
            raw_pred = self.sess.run(None, {self.input_name: features})[0]
            self.pred_history.append(raw_pred[0][1])
            smooth_pred = np.mean(self.pred_history)
            
            # --- Adaptive Logic ---
            base_threshold = 0.015 # Increased significantly
            dynamic_threshold = base_threshold * (1.0 + (volatility / 20.0))
            dynamic_stop_pct = min(0.008, 0.003 * (1.0 + (volatility / 15.0)))
            
            if smooth_pred > dynamic_threshold and self.position == 0:
                risk_size = self.usdt * 0.9
                self.position = risk_size / current_price
                self.entry_price = current_price
                self.usdt -= risk_size
                self.log(f"🚀 [ENTRY] @ {current_price:.2f} | Pred(S): {smooth_pred:.4f} | Stop: {dynamic_stop_pct*100:.2f}%")
            
            elif smooth_pred < -dynamic_threshold * 0.5 and self.position > 0:
                sale_usdt = self.position * current_price
                self.usdt += sale_usdt
                self.position = 0
                self.equity = self.usdt
                self.log(f"📉 [EXIT] @ {current_price:.2f} | Equity: {self.equity:.2f} | Pred(S): {smooth_pred:.6f}")

            if self.position > 0:
                self.equity = self.usdt + (self.position * current_price)
                if current_price < self.entry_price * (1.0 - dynamic_stop_pct):
                     sale_usdt = self.position * current_price
                     self.usdt += sale_usdt
                     self.log(f"🛑 [STOP] @ {current_price:.2f} | Loss: -{dynamic_stop_pct*100:.2f}% | Equity: {self.equity:.2f}")
                     self.position = 0
                     self.equity = self.usdt
                elif current_price > self.entry_price * 1.0015:
                     if current_price < self.entry_price * 1.0005:
                         sale_usdt = self.position * current_price
                         self.usdt += sale_usdt
                         self.log(f"⚖️ [BE+] @ {current_price:.2f} | Equity: {self.equity:.2f}")
                         self.position = 0
                         self.equity = self.usdt
        except Exception:
            pass

if __name__ == "__main__":
    trader = OptimizedChallengeTraderV3()
    print("🌌 Singularity V3 Adaptive Trader Active...", flush=True)
    while True:
        trader.run_cycle()
        time.sleep(1)
