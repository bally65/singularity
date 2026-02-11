import onnxruntime as ort
import pandas as pd
import numpy as np
import torch

def validate():
    # Load data
    df = pd.read_csv('dataset.csv').apply(pd.to_numeric, errors='coerce').dropna()
    if len(df) < 100:
        print("❌ Not enough data for validation.")
        return

    feature_cols = ['velocity', 'accel', 'entropy', 'mass', 'imbalance', 'liq_force']
    data = df[feature_cols].values
    labels = df['label_return_60s'].values

    # Normalize (using same simple logic as training)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-6)

    # Prepare sessions
    sess = ort.InferenceSession('singularity_v3.onnx')
    input_name = sess.get_inputs()[0].name

    correct_direction = 0
    total = 0
    
    # Take the last 500 samples for validation
    test_range = range(len(data) - 560, len(data) - 60)
    if len(data) < 560: test_range = range(0, len(data) - 60)

    for i in test_range:
        seq = data[i:i+60].astype(np.float32).reshape(1, 60, 6)
        pred = sess.run(None, {input_name: seq})[0]
        # pred is [Q10, Q50, Q90]. Use Q50 (index 1)
        predicted_return = pred[0][1]
        actual_return = labels[i+59]

        if (predicted_return > 0 and actual_return > 0) or (predicted_return < 0 and actual_return < 0):
            correct_direction += 1
        total += 1

    accuracy = (correct_direction / total) * 100 if total > 0 else 0
    print(f"📊 Singularity V3 Validation Results:")
    print(f"   - Samples tested: {total}")
    print(f"   - Directional Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    validate()
