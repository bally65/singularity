package core

import (
	"fmt"
	"os"
	"sync"
	"time"
)

type Backtester struct {
	mu        sync.RWMutex
	portfolio PortfolioState
	trades    []ExecutedOrder
}

func NewBacktester(initialBalance float64) *Backtester {
	return &Backtester{
		portfolio: PortfolioState{
			Balance:   initialBalance,
			Positions: make(map[string]float64),
			Equity:    initialBalance,
		},
		trades: make([]ExecutedOrder, 0),
	}
}

// ExecutePaperTrade handles simulated execution
func (b *Backtester) ExecutePaperTrade(symbol string, side Direction, price float64, qty float64) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	cost := price * qty
	sideStr := "BUY"
	if side == Buy {
		if b.portfolio.Balance < cost {
			return fmt.Errorf("insufficient balance: have %.2f, need %.2f", b.portfolio.Balance, cost)
		}
		b.portfolio.Balance -= cost
		b.portfolio.Positions[symbol] += qty
	} else {
		sideStr = "SELL"
		if b.portfolio.Positions[symbol] < qty {
			return fmt.Errorf("insufficient position: have %.4f, need %.4f", b.portfolio.Positions[symbol], qty)
		}
		b.portfolio.Balance += cost
		b.portfolio.Positions[symbol] -= qty
	}

	// Simple Equity calculation (Cash + Position Value)
	posValue := 0.0
	for sym, q := range b.portfolio.Positions {
		if sym == symbol {
			posValue += q * price
		}
	}
	b.portfolio.Equity = b.portfolio.Balance + posValue

	order := ExecutedOrder{
		OrderID:   fmt.Sprintf("P-%d", time.Now().UnixNano()),
		Symbol:    symbol,
		Side:      side,
		Price:     price,
		Qty:       qty,
		Status:    OrderFilled,
		Timestamp: time.Now().UnixMicro(),
	}
	b.trades = append(b.trades, order)
	
	// Persistent Challenge Log
	logLine := fmt.Sprintf("[%s] %s %s %.6f @ %.2f | Bal: %.2f | Equity: %.2f\n", 
		time.Now().Format("2006-01-02 15:04:05"), sideStr, symbol, qty, price, b.portfolio.Balance, b.portfolio.Equity)
	
	fmt.Print(" [BACKTEST] " + logLine)
	
	f, err := os.OpenFile("challenge_trades.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err == nil {
		f.Write([]byte(logLine))
		f.Close()
	}

	return nil
}

func (b *Backtester) GetStats() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return fmt.Sprintf("Balance: %.2f | Equity: %.2f | Trades: %d", b.portfolio.Balance, b.portfolio.Equity, len(b.trades))
}
