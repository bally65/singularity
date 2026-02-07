# Singularity Defense Model Training Guide

## 1. Environment Setup
On your GPU server, ensure you have Python 3.8+ and CUDA installed.

Install dependencies:
```bash
pip install torch pandas numpy
```

## 2. Data Transfer
Make sure to upload your `dataset.csv` to the same directory as `train_transformer.py`.
If you are starting fresh, you might need to run the Go engine to collect data first, or upload your existing `dataset.csv`.

## 3. Running Training
Run the training script (supports GPU automatically):

```bash
python train_transformer.py
```

### Configuration
You can edit the top section of `train_transformer.py` to adjust hyperparameters:
- `epochs`: Default is 20. Increase for better results on a powerful server.
- `batch_size`: Default is 64. Increase to 128 or 256 if you have a large VRAM GPU (e.g. A100/H100).
- `seq_length`: Currently 60 (approx 1 minute context).

## 4. Output
The script will generate:
- `singularity_v2_defense.pth`: PyTorch Checkpoint.
- `singularity_defense.onnx`: ONNX model for Go inference.

Download these files back to your local machine to run the Go strategy engine.
