import torch
import torch.nn as nn
import math

# --- Model Definition (Must match training) ---
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

class SingularityTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(SingularityTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model * 30, 1) # hardcoded seq_len=30

    def forward(self, src):
        src = src.permute(1, 0, 2) 
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2).reshape(output.size(1), -1)
        prediction = self.decoder(output)
        return prediction.squeeze()

# --- Export Logic ---
def export():
    input_dim = 5 # velocity, accel, entropy, mass, imbalance
    seq_length = 30
    
    print("Loading model...")
    model = SingularityTransformer(input_dim=input_dim)
    model.load_state_dict(torch.load("singularity_v1.pth"))
    model.eval()
    
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, seq_length, input_dim)
    
    # Export with dynamic axes if we want batching support, but fixed for now is safer for C++
    torch.onnx.export(model, dummy_input, "singularity.onnx", 
                      input_names=['input'], 
                      output_names=['output'],
                      verbose=False)
    print("Done! singularity.onnx created.")

if __name__ == "__main__":
    export()
