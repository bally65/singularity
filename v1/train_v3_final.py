# === Singularity V3 一鍵訓練程式碼 (最終相容性修復版) ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def train_v3():
    # --- 1. 數據抓取與合併 ---
    if not os.path.exists('dataset.csv'):
        print("🚀 正在從 GitHub 抓取數據塊...")
        # Note: In Colab use !wget, in script use os.system
        os.system("wget -q https://raw.githubusercontent.com/bally65/singularity/master/dataset.csv.partaa")
        os.system("wget -q https://raw.githubusercontent.com/bally65/singularity/master/dataset.csv.partab")
        os.system("wget -q https://raw.githubusercontent.com/bally65/singularity/master/dataset.csv.partac")
        os.system("wget -q https://raw.githubusercontent.com/bally65/singularity/master/dataset.csv.partad")
        print("📂 正在合併數據塊...")
        os.system("cat dataset.csv.part* > dataset.csv")
    else:
        print("📦 數據檔案已存在，跳過抓取。")

    # --- 2. 數據讀取與清理 ---
    print("📊 正在讀取數據 (自動修復接縫錯誤)...")
    try:
        # 使用 on_bad_lines='skip' 處理分割產生的殘缺行
        df = pd.read_csv('dataset.csv', on_bad_lines='skip', low_memory=False).dropna()
        print(f"✅ 成功讀取 {len(df)} 筆有效數據！")
    except Exception as e:
        print(f"❌ 讀取失敗: {e}"); return

    # --- 3. 超參數設定 ---
    SEQ_LENGTH = 60
    BATCH_SIZE = 64
    EPOCHS = 30
    FEATURE_COLS = ['velocity', 'accel', 'entropy', 'mass', 'imbalance', 'liq_force']
    TARGET_COL = 'label_return_60s'

    # --- 4. 模型架構: CNN + LSTM + Attention ---
    class SingularityV3Model(nn.Module):
        def __init__(self, input_dim, d_model=128, nhead=8):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
            self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=0.1)
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
            self.head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            x = F.relu(self.conv1(x))
            x = x.transpose(1, 2)
            lstm_out, _ = self.lstm(x)
            attn_in = lstm_out.transpose(0, 1) # (seq, batch, dim)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in)
            return self.head(attn_out[-1])

    # --- 5. 數據預處理 (標準化) ---
    data = df[FEATURE_COLS].values
    labels = df[TARGET_COL].values
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-6)
    
    X, Y = [], []
    for i in range(len(data) - SEQ_LENGTH):
        X.append(data[i:i+SEQ_LENGTH])
        Y.append(labels[i+SEQ_LENGTH-1])
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        np.array(X), np.array(Y), test_size=0.1, shuffle=False
    )

    # --- 6. 訓練循環 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingularityV3Model(len(FEATURE_COLS)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.HuberLoss()

    print(f"🔥 開始在 {device} 上訓練 V3 模型...")
    for epoch in range(EPOCHS):
        model.train()
        idx = np.random.choice(len(X_train), BATCH_SIZE)
        x_batch = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
        y_batch = torch.tensor(Y_train[idx], dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        pred = model(x_batch)[:, 1]
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

    # --- 7. ONNX 模型導出 (相容性修復區) ---
    print("📦 正在以相容模式導出 ONNX 模型...")
    model.eval()
    model.to("cpu") # 關鍵：切換回 CPU 導出以避免 FakeTensor 指針錯誤
    dummy = torch.randn(1, SEQ_LENGTH, len(FEATURE_COLS)).to("cpu")

    try:
        torch.onnx.export(
            model,
            dummy,
            "singularity_v3.onnx",
            export_params=True,
            opset_version=14, # 支援 MultiheadAttention 的穩定版本
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            training=torch.onnx.TrainingMode.EVAL
        )
        print(" " + "—"*30)
        print("✅ 任務完全成功！")
        print("📦 檔案 singularity_v3.onnx 已生成，請在資料夾下載。")
        print("—"*30)
    except Exception as e:
        print(f"❌ 導出依然遇到問題: {e}")

if __name__ == "__main__":
    train_v3()
