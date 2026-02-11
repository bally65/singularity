import pandas as pd
import numpy as np
import os

    def convert_v2_to_v3_format(v2_path, v3_path):
    if not os.path.exists(v2_path):
        return
    
    # Read V2 raw data
    df = pd.read_csv(v2_path, names=['ts_now', 'ts_event', 'price', 'qty', 'maker'], 
                         on_bad_lines='skip', engine='python')
    
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df = df.dropna(subset=['price', 'qty'])

    # Features
    df['price_change'] = df['price'].diff().fillna(0)
    df['velocity'] = df['price_change']
    df['accel'] = df['velocity'].diff().fillna(0)
    
    # --- Optimization: Real Entropy Calculation ---
    # Rolling standard deviation of price changes as a proxy for entropy/volatility
    df['entropy'] = df['price_change'].rolling(window=100).std().fillna(0)
    
    df['mass'] = df['qty'].rolling(window=20).sum().fillna(0)
    df['imbalance'] = np.where(df['maker'] == True, -1, 1) * df['qty']
    df['liq_force'] = 0
    
    # Target
    df['label_return_60s'] = df['price'].shift(-600) / df['price'] - 1
    df['label_return_60s'] = df['label_return_60s'].ffill().fillna(0)
    
    # Capping outliers to avoid inf/nan in training
    for col in ['velocity', 'accel', 'mass', 'imbalance', 'label_return_60s']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        q_low = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df[col] = df[col].clip(q_low, q_high)

    v3_df = df[['velocity', 'accel', 'entropy', 'mass', 'imbalance', 'liq_force', 'label_return_60s']]
    v3_df.to_csv(v3_path, index=False)

    print(f"✅ Re-converted {len(v3_df)} rows to {v3_path}")

if __name__ == "__main__":
    convert_v2_to_v3_format('/home/aa598/.openclaw/workspace/singularity/project/v2/dataset_v2.csv', '/home/aa598/.openclaw/workspace/singularity/project/v1/dataset.csv')
