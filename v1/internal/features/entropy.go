package features

import (
	"math"
	"singularity/internal/core"
)

// EntropyEngine adds Information Theory metrics
type EntropyEngine struct {
	// Histogram configuration
	numBins int
}

func NewEntropyEngine() *EntropyEngine {
	return &EntropyEngine{
		numBins: 20, // Divide order book into 20 structural bins
	}
}

// CalculateBookEntropy computes Shannon Entropy (H) of the Order Book volume profile.
// High Entropy = Uniform distribution (Chaos / Uncertainty)
// Low Entropy = Concentrated liquidity (Trend / Structure)
// H(X) = -sum(p(x) * log2(p(x)))
func (e *EntropyEngine) CalculateBookEntropy(book *core.OrderBookL2) float64 {
	if len(book.Bids) == 0 || len(book.Asks) == 0 {
		return 0.0
	}

	// 1. Aggregation: Build a probability distribution of liquidity
	// We pool top N levels from both sides
	totalVol := 0.0
	depth := 20
	if len(book.Bids) < depth { depth = len(book.Bids) }
	
	volumes := make([]float64, 0, depth*2)
	
	// Bids
	for i := 0; i < depth; i++ {
		vol := book.Bids[i].Size
		volumes = append(volumes, vol)
		totalVol += vol
	}
	// Asks
	for i := 0; i < depth; i++ {
		if i >= len(book.Asks) { break }
		vol := book.Asks[i].Size
		volumes = append(volumes, vol)
		totalVol += vol
	}

	if totalVol == 0 {
		return 0.0
	}

	// 2. Compute Entropy
	entropy := 0.0
	for _, v := range volumes {
		if v > 0 {
			p := v / totalVol
			entropy -= p * math.Log2(p)
		}
	}
	
	// Normalize? Max entropy for N items is Log2(N).
	// Let's return raw bits for now. Max bits = Log2(40) ~= 5.32 bits
	return entropy
}
