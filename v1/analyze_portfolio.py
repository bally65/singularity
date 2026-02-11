import pandas as pd
import numpy as np
from datetime import datetime

log_path = '/home/aa598/.openclaw/workspace/singularity/project/v1/challenge_9000.log'

def analyze():
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    trades = []
    initial_balance = 281.25
    current_equity = initial_balance
    
    for line in lines:
        if 'BUY' in line:
            parts = line.split('|')
            price = float(parts[0].split('@')[1].strip())
            trades.append({'type': 'BUY', 'price': price, 'time': line[1:20]})
        elif 'SELL' in line:
            parts = line.split('|')
            price = float(parts[0].split('@')[1].strip())
            equity = float(parts[1].split(':')[1].strip())
            trades.append({'type': 'SELL', 'price': price, 'equity': equity, 'time': line[1:20]})
            current_equity = equity

    total_trades = len([t for t in trades if t['type'] == 'SELL'])
    profit_loss = current_equity - initial_balance
    roi = (profit_loss / initial_balance) * 100
    
    print(f"--- Singularity V3 盈虧分析報告 ---")
    print(f"初始資金: {initial_balance} USDT")
    print(f"目前淨值: {current_equity:.2f} USDT")
    print(f"總盈虧: {profit_loss:+.2f} USDT ({roi:+.2f}%)")
    print(f"已完成交易對: {total_trades}")
    
    if total_trades > 0:
        win_trades = 0
        last_buy = 0
        for t in trades:
            if t['type'] == 'BUY': last_buy = t['price']
            if t['type'] == 'SELL' and t['price'] > last_buy: win_trades += 1
        
        print(f"勝率: {(win_trades/total_trades)*100:.1f}%")

if __name__ == "__main__":
    analyze()
