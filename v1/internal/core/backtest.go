package core

import (
	"fmt"
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
	if side == Buy {
		if b.portfolio.Balance < cost {
			return fmt.Errorf("insufficient balance: have %.2f, need %.2f", b.portfolio.Balance, cost)
		}
		b.portfolio.Balance -= cost
		b.portfolio.Positions[symbol] += qty
	} else {
		if b.portfolio.Positions[symbol] < qty {
			return fmt.Errorf("insufficient position: have %.4f, need %.4f", b.portfolio.Positions[symbol], qty)
		}
		b.portfolio.Balance += cost
		b.portfolio.Positions[symbol] -= qty
	}

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
	return nil
}

func (b *Backtester) GetStats() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return fmt.Sprintf("Balance: %.2f | Trades: %d", b.portfolio.Balance, len(b.trades))
}
