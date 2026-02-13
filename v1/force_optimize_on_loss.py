import os
import json
import subprocess

# If equity drops below 280 (loss of 1.25 USDT), trigger re-training
CHALLENGE_LOG = "/home/aa598/.openclaw/workspace/singularity/project/v1/challenge_9000.log"

def check_and_force():
    if not os.path.exists(CHALLENGE_LOG): return
    with open(CHALLENGE_LOG, 'r') as f:
        last_line = f.readlines()[-1]
        if "Equity:" in last_line:
            equity = float(last_line.split("Equity:")[1].split("|")[0].strip())
            if equity < 280.0:
                print("🚨 Loss detected! Triggering Emergency Self-Optimization...")
                subprocess.run(["python3", "self_optimize.py"], cwd="/home/aa598/.openclaw/workspace/singularity/project/v1")

if __name__ == "__main__":
    check_and_force()
