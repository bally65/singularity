#!/bin/bash
# Molty's Singularity Autopilot Runner

echo "🎨 Building Singularity Engine..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
go build -o engine cmd/engine/main.go

if [ ! -f "libonnxruntime.so" ]; then
    echo "⚠️  Warning: libonnxruntime.so not found in project root."
    echo "AI Inference will be disabled until you provide the shared library."
fi

echo "🚀 Starting Engine..."
echo "To enable Discord alerts, run with: DISCORD_WEBHOOK=your_url ./engine"
./engine
