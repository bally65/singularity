import pandas as pd
import shutil
import os
import time

def analyze_dataset():
    src_file = "dataset.csv"
    temp_file = "dataset_temp.csv"

    print(f"Loading {src_file}...")

    # Copy to temp file to avoid locking issues with the running engine
    try:
        shutil.copy2(src_file, temp_file)
    except Exception as e:
        print(f"Error copying file: {e}")
        return

    try:
        # Load CSV
        # columns: timestamp, price, velocity, accel, entropy, mass, imbalance, label_return_60s, label_class_60s
        df = pd.read_csv(temp_file)
        
        print("\n" + "="*40)
        print(f"📊 DATASET HEALTH CHECK (N={len(df)})")
        print("="*40)

        # 1. Class Distribution
        print("\n[Labels Distribution]")
        # label_class_60s: 1 (Up), -1 (Down), 0 (Flat)
        counts = df['label_class_60s'].value_counts()
        total = len(df)
        
        for cls in [1, -1, 0]:
            count = counts.get(cls, 0)
            pct = (count / total) * 100
            print(f"  Trace {cls:>2}: {count:>6} rows ({pct:.2f}%)")

        # 2. Return Statistics (Volatility)
        print("\n[Return Statistics (60s)]")
        # label_return_60s is the pct change
        returns = df['label_return_60s']
        print(f"  Mean Return: {returns.mean():.6f}")
        print(f"  Std Dev (Vol): {returns.std():.6f}")
        print(f"  Max Up:      {returns.max():.6f}")
        print(f"  Max Down:    {returns.min():.6f}")

        # 3. Physics Feature Check
        print("\n[Physics Check]")
        print(f"  Max Velocity: {df['velocity'].max():.4f}")
        print(f"  Max Accel:    {df['accel'].max():.4f}")
        print(f"  Avg Entropy:  {df['entropy'].mean():.4f}")

        print("\n" + "="*40)
        print("✅ Analysis Complete.")

    except Exception as e:
        print(f"Analysis failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    if not os.path.exists("dataset.csv"):
        print("dataset.csv not found!")
    else:
        analyze_dataset()
