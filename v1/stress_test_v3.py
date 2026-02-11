import onnxruntime as ort
import pandas as pd
import numpy as np
import os
from io import StringIO

# Configuration
V2_DATA_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v2/dataset_v2.csv'
MODEL_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v1/singularity_v3.onnx'

def backtest_adaptive(df_test):
    sess = ort.InferenceSession(MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    
    initial_usdt = 281.25
    usdt = initial_usdt
    position = 0.0
    entry_price = 0.0
    equity_curve = []
    trades = 0
    
    # Feature calculation for the whole slice
    df = df_test.copy()
    df['price_change'] = df['price'].diff().fillna(0)
    df['velocity'] = df['price_change']
    df['accel'] = df['velocity'].diff().fillna(0)
    df['entropy'] = df['price_change'].rolling(window=50).std().fillna(0)
    df['mass'] = df['qty'].rolling(window=20).mean().fillna(0)
    df['imbalance'] = np.where(df['maker'] == True, -1, 1) * df['qty']
    df['liq_force'] = 0
    
    feature_cols = ['velocity', 'accel', 'entropy', 'mass', 'imbalance', 'liq_force']
    
    print(f"📊 Starting Adaptive Stress Test on {len(df)} samples...")
    
    # Simple loop for simulation
    for i in range(150, len(df)):
        current_price = df['price'].iloc[i]
        volatility = df['price'].iloc[i-100:i].std() if i > 100 else 0
        
        # Prepare features
        feat_window = df[feature_cols].iloc[i-60:i].values
        feat_window = (feat_window - np.mean(feat_window, axis=0)) / (np.std(feat_window, axis=0) + 1e-6)
        seq = feat_window.astype(np.float32).reshape(1, 60, 6)
        
        # Pred
        raw_pred = sess.run(None, {input_name: seq})[0][0][1]
        
        # Logic
        base_threshold = 0.001
        dynamic_threshold = base_threshold * (1.0 + (volatility / 30.0))
        dynamic_stop_pct = min(0.006, 0.002 * (1.0 + (volatility / 20.0)))

        if raw_pred > dynamic_threshold and position == 0:
            position = usdt * 0.9 / current_price
            entry_price = current_price
            usdt -= (usdt * 0.9)
            trades += 1
            # print(f"   Trade {trades}: BUY @ {current_price:.2f}")
        elif raw_pred < -dynamic_threshold * 0.5 and position > 0:
            usdt += position * current_price
            # print(f"   Trade {trades}: EXIT (SIGNAL) @ {current_price:.2f} | PnL: {current_price - entry_price:.2f}")
            position = 0
        elif position > 0:
            if current_price < entry_price * (1.0 - dynamic_stop_pct):
                usdt += position * current_price
                # print(f"   Trade {trades}: EXIT (STOP) @ {current_price:.2f} | PnL: {current_price - entry_price:.2f}")
                position = 0
            elif current_price > entry_price * 1.0015:
                if current_price < entry_price * 1.0005:
                    usdt += position * current_price
                    # print(f"   Trade {trades}: EXIT (BE) @ {current_price:.2f} | PnL: {current_price - entry_price:.2f}")
                    position = 0
        
        equity = usdt + (position * current_price)
        equity_curve.append(equity)

    final_equity = equity_curve[-1] if equity_curve else initial_usdt
    print(f"✅ Stress Test Result:")
    print(f"   Final Equity: {final_equity:.2f} USDT")
    print(f"   Net PnL: {final_equity - initial_usdt:+.2f} USDT")
    print(f"   Total Trades: {trades}")

if __name__ == "__main__":
    # Load last 10000 rows
    raw_tail = os.popen(f'tail -n 10000 {V2_DATA_PATH}').read()
    from io import StringIO
    df = pd.read_csv(StringIO(raw_tail), names=['ts_now', 'ts_event', 'price', 'qty', 'maker'], on_bad_lines='skip', engine='python').apply(pd.to_numeric, errors='coerce').dropna()
    df = df[df['price'] > 0]
    backtest_adaptive(df)
