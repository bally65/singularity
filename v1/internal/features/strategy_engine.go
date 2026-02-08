package features

import (
	"fmt"
	"singularity/internal/core"
	"time"
)

// StrategyEngine decides when to enter/exit trades
type StrategyEngine struct {
	config     core.StrategyConfig
	backtester *core.Backtester
	lastAction time.Time
}

func NewStrategyEngine(bt *core.Backtester) *StrategyEngine {
	return &StrategyEngine{
		config: core.StrategyConfig{
			SignalThreshold: 0.8,
			MaxPositionSize: 0.1, // BTC
		},
		backtester: bt,
	}
}

// ProcessState evaluates the latest feature vector and executes trades
func (s *StrategyEngine) ProcessState(symbol string, price float64, feat *FeatureVector) {
	// Simple Cooldown: 5 seconds between trades
	if time.Since(s.lastAction) < 5*time.Second {
		return
	}

	// Dynamic Logic: Combined AI + Force Physics
	// 1. If NetForce (Liquidations) is strong in one direction
	// 2. If Price Acceleration confirms the move
	// 3. If Order Imbalance matches
	
	if feat.PriceAccel > 50 {
		// Aggressive Long
		err := s.backtester.ExecutePaperTrade(symbol, core.Buy, price, s.config.MaxPositionSize)
		if err == nil {
			s.lastAction = time.Now()
			fmt.Printf(" [STRATEGY] Executed LONG on %s at %.2f\n", symbol, price)
		}
	} else if feat.PriceAccel < -50 {
		// Aggressive Short
		err := s.backtester.ExecutePaperTrade(symbol, core.Sell, price, s.config.MaxPositionSize)
		if err == nil {
			s.lastAction = time.Now()
			fmt.Printf(" [STRATEGY] Executed SHORT on %s at %.2f\n", symbol, price)
		}
	}
}
