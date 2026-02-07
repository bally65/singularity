import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import os
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
csv_file = 'dataset.csv'
seq_length = 60           # Increased context window for LSTM memory (approx 1 min if 1s ticks)
batch_size = 64
learning_rate = 0.0005
epochs = 20
feature_cols = ['velocity', 'accel', 'entropy', 'mass', 'imbalance', 'liq_force']
target_col = 'label_return_60s' 

# Defense Prediction Settings
quantile_levels = [0.1, 0.5, 0.9] # Predict 10th (VaR risk), 50th (Median), 90th (Upside) percentiles

# ==========================================
# 2. Dataset Loader (Enhanced)
# ==========================================
class FinancialDataset(Dataset):
    def __init__(self, csv_path, seq_len):
        print(f"Loading data from {csv_path}...")
        try:
            self.df = pd.read_csv(csv_path)
            self.df = self.df.dropna()
        except FileNotFoundError:
            print("Error: dataset.csv not found.")
            self.df = pd.DataFrame()

        self.seq_len = seq_len
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.labels = self.df[target_col].values.astype(np.float32)
        
        # Robust Scaling (Median/IQR) to handle outliers better than Z-Score
        self.median = np.median(self.features, axis=0)
        q75, q25 = np.percentile(self.features, [75 ,25], axis=0)
        self.iqr = q75 - q25 + 1e-6
        self.features = (self.features - self.median) / self.iqr

    def __len__(self):
        return max(0, len(self.df) - self.seq_len)

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len].astype(np.float32)
        y = self.labels[idx + self.seq_len - 1].astype(np.float32)
        return torch.tensor(x), torch.tensor(y)

# ==========================================
# 3. Hybrid Defense Model (LSTM-mTrans-MLP)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class SingularityDefenseModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, lstm_hidden=64):
        super(SingularityDefenseModel, self).__init__()
        
        # --- Stage 1: LSTM (Temporal Context & Noise Filtering) ---
        # Input: [Batch, SeqLen, Features]
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, 
                            num_layers=1, batch_first=True, bidirectional=False)
        
        # Adapter to project LSTM output to Transformer d_model
        self.lstm_projection = nn.Linear(lstm_hidden, d_model)
        
        # --- Stage 2: Transformer (Regime Detection & Global Attention) ---
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # --- Stage 3: MLP Heads (Probabilistic Prediction) ---
        # We predict 3 quantiles for risk management: 10% (Risk), 50% (Price), 90% (Opportunity)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model * seq_length, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3) # Output: [q10, q50, q90]
        )

    def forward(self, x):
        # x: [Batch, SeqLen, Features]
        
        # 1. LSTM Pass
        lstm_out, _ = self.lstm(x) # -> [Batch, SeqLen, lstm_hidden]
        
        # 2. Project to Transformer Dim
        x_proj = self.lstm_projection(lstm_out) # -> [Batch, SeqLen, d_model]
        
        # 3. Transformer Pass (requires [SeqLen, Batch, d_model])
        x_trans = x_proj.permute(1, 0, 2)
        x_trans = self.pos_encoder(x_trans)
        trans_out = self.transformer_encoder(x_trans) # -> [SeqLen, Batch, d_model]
        
        # 4. Flatten & MLP
        # [Batch, SeqLen * d_model]
        flat_out = trans_out.permute(1, 0, 2).reshape(trans_out.size(1), -1)
        prediction = self.mlp_head(flat_out)
        
        return prediction # [Batch, 3]

# ==========================================
# 4. Quantile Loss Function (For Risk Ranges)
# ==========================================
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        loss = 0
        target = target.unsqueeze(1) # [Batch, 1]
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i+1]
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss

# ==========================================
# 5. Training Loop
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    print(f"Model Architecture: LSTM -> Transformer -> MLP (Quantile Regression)")

    dataset = FinancialDataset(csv_file, seq_length)
    if len(dataset) < 100:
        print("Not enough data to train!")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SingularityDefenseModel(input_dim=len(feature_cols)).to(device)
    
    if os.path.exists("singularity_v2_defense.pth"):
        print("Resuming training from singularity_v2_defense.pth...")
        try:
            model.load_state_dict(torch.load("singularity_v2_defense.pth"))
        except:
            print("Checkpoint mismatch, starting clean.")

    criterion = QuantileLoss(quantile_levels)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Quantile Loss: {total_loss/(batch_idx+1):.6f}")

    # Save
    torch.save(model.state_dict(), "singularity_v2_defense.pth")
    print("[OK] Model saved to singularity_v2_defense.pth")
    
    # Export ONNX
    model.eval() # Set to eval mode before export
    dummy_input = torch.randn(1, seq_length, len(feature_cols)).to(device)
    torch.onnx.export(model, dummy_input, "singularity_defense.onnx", 
                      input_names=['input'], output_names=['quantiles_10_50_90'],
                      verbose=False)
    print("\n[OK] Exported 'singularity_defense.onnx'")
    print("   Output format: [Batch, 3] -> (10th percentile, Median, 90th percentile)")
    print("   Use 10th percentile for VaR (Risk Control)")
    print("   Use 50th percentile for Price Prediction")

if __name__ == "__main__":
    train()
