# 🌌 Singularity Prediction Engine

A high-frequency trading (HFT) infrastructure combining **Quantum Physics heuristics** with **Transformer-based AI** to predict and execute trades on crypto-currency futures.

## 🚀 Key Features

- **Hybrid Analysis**: Combines real-time Kinematics (Velocity/Accel) with Entropy analysis and ML.
- **Liquidation Force Detection**: Tracks real-time liquidation events as an external "Force" acting on price.
- **AI Core**: PyTorch-trained LSTM-Transformer models exported via ONNX for low-latency Go inference.
- **Auto-Backtester**: Integrated paper trading engine for strategy validation without financial risk.
- **Cyberpunk Dashboard**: Real-time WebSocket-driven UI for monitoring price action, physics metrics, and portfolio balance.
- **Discord Integration**: Automated alerts sent directly to your Discord channel via Webhooks.

## 🛠 Project Structure

- `/cmd/engine`: Main Go entry point.
- `/internal/features`: Feature engineering, ONNX inference, and Strategy logic.
- `/internal/adapter`: Exchange connectivity (Binance Futures WebSocket).
- `/internal/core`: Core types, backtesting logic, and strategy definitions.
- `/web`: Frontend dashboard.
- `train_transformer.py`: Python training script for the AI model.

## 🏁 Getting Started

1. **Environment**: Ensure you have Go 1.25+ and Python 3.11+.
2. **Setup**:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```
3. **Alerts**: Set your Discord Webhook:
   ```bash
   export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
   ./engine
   ```

## 🧠 Development Status: [ACTIVE]
Managed by **Molty (Digital Butler)** via OpenClaw. Continuous integration and feature expansion in progress.
