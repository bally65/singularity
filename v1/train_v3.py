import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Hyperparameters ---
SEQ_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
FEATURE_COLS = ['velocity', 'accel', 'entropy', 'mass', 'imbalance', 'liq_force']
TARGET_COL = 'label_return_60s'

class FinancialDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class SingularityV3Model(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=0.1)
        self.attention = MultiHeadAttention(d_model, nhead)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # [Q10, Q50, Q90]
        )

    def forward(self, x):
        # x: [Batch, SeqLen, Features]
        x = x.transpose(1, 2) # -> [Batch, Features, SeqLen]
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2) # -> [Batch, SeqLen, d_model]
        
        lstm_out, _ = self.lstm(x)
        attn_in = lstm_out.transpose(0, 1) # [SeqLen, Batch, d_model]
        attn_out = self.attention(attn_in)
        last_state = attn_out[-1] # [Batch, d_model]
        return self.head(last_state)

def prepare_data(csv_path):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path, low_memory=False).apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 500: return None, None
    
    data = df[FEATURE_COLS].values
    labels = df[TARGET_COL].values
    
    # Simple normalization
    data = (data - np.nanmean(data, axis=0)) / (np.nanstd(data, axis=0) + 1e-6)
    
    X, Y = [], []
    for i in range(len(data) - SEQ_LENGTH):
        X.append(data[i:i+SEQ_LENGTH])
        Y.append(labels[i+SEQ_LENGTH-1])
    
    return np.array(X), np.array(Y)

def train(resume=False):
    X, Y = prepare_data('dataset.csv')
    if X is None:
        print("❌ Not enough data in dataset.csv to train V3.")
        return

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)
    
    train_loader = DataLoader(FinancialDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FinancialDataset(X_val, Y_val), batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingularityV3Model(len(FEATURE_COLS)).to(device)
    
    if resume and os.path.exists("singularity_v3_latest.pth"):
        model.load_state_dict(torch.load("singularity_v3_latest.pth"))
        print("🔄 Resuming from existing weights...")

    # Load adaptive config if exists
    config_path = "training_config.json"
    actual_lr = LEARNING_RATE
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            actual_lr = json.load(f).get("learning_rate", LEARNING_RATE)
            print(f"🧠 Using Adaptive Learning Rate: {actual_lr:.6f}")

    optimizer = optim.AdamW(model.parameters(), lr=actual_lr)
    criterion = nn.HuberLoss() # More robust to outliers in price data

    print(f"🚀 Training V3 on {device}...")
    epochs_to_run = 5 if resume else EPOCHS # Short fine-tune if iterating
    
    for epoch in range(epochs_to_run):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)[:, 1] # Predict median (q50)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs_to_run} | Loss: {total_loss/len(train_loader):.6f}", flush=True)

    # Save weights for next iteration
    torch.save(model.state_dict(), "singularity_v3_latest.pth")

    # Export
    model.eval()
    dummy = torch.randn(1, SEQ_LENGTH, len(FEATURE_COLS)).to(device)
    torch.onnx.export(model, dummy, "singularity_v3.onnx", input_names=['input'], output_names=['output'])
    print("✅ V3 Model Optimized and Exported to singularity_v3.onnx")

if __name__ == "__main__":
    import sys
    do_resume = "--resume" in sys.argv
    train(resume=do_resume)
