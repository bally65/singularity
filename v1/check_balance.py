import onnxruntime as ort
import pandas as pd
import numpy as np
import os

V2_DATA_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v2/dataset_v2.csv'
MODEL_PATH = '/home/aa598/.openclaw/workspace/singularity/project/v1/singularity_v3.onnx'

def check():
    df = pd.read_csv(V2_DATA_PATH, names=['ts_now', 'ts_event', 'price', 'qty', 'maker'], on_bad_lines='skip', engine='python').apply(pd.to_numeric, errors='coerce').dropna()
    current_price = df['price'].iloc[-1]
    
    df['price_change'] = df['price'].diff().fillna(0)
    df['velocity'] = df['price_change']
    df['accel'] = df['velocity'].diff().fillna(0)
    df['entropy'] = df['price_change'].rolling(window=50).std().fillna(0)
    df['mass'] = df['qty'].rolling(window=20).sum().fillna(0)
    df['imbalance'] = np.where(df['maker'] == True, -1, 1) * df['qty']
    df['liq_force'] = 0
    
    features = df[['velocity', 'accel', 'entropy', 'mass', 'imbalance', 'liq_force']].tail(60).values
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    seq = features.astype(np.float32).reshape(1, 60, 6)
    
    sess = ort.InferenceSession(MODEL_PATH)
    pred = sess.run(None, {sess.get_inputs()[0].name: seq})[0]
    
    print(f"--- Singularity Status ---")
    print(f"Current BTC Price: {current_price:.2f}")
    print(f"V3 Model Prediction (Median): {pred[0][1]:.6f}")
    
    if os.path.exists('challenge_9000.log'):
        with open('challenge_9000.log', 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"Last Transaction: {lines[-1].strip()}")

if __name__ == "__main__":
    check()
