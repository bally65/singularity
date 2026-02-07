package features

import (
	"fmt"
	"math"
	"singularity/internal/core"
	"time"
)

// FeatureVector represents the calculated state at a specific point in time
type FeatureVector struct {
	Timestamp      int64
	PriceVelocity  float64 
	PriceAccel     float64 
	OrderImbalance float64 
	Entropy        float64 
	
	// New Physics Props
	MassResistance float64 
	NetForce       float64 
	WhaleForce     float64 // Net volume of large "whale" trades
}

// Prediction represents a Gaussian PDF of future price
type Prediction struct {
	Mean  float64
	Sigma float64
	Source string // "AI" or "Heuristic"
}

// Engine manages the state and calculation of features
type Engine struct {
	windowSize time.Duration
	entropy    *EntropyEngine 
	model      *Model
	backtester *core.Backtester
	
	// State for Derivatives
	lastPrice       float64
	lastPriceTime   int64
	lastVelocity    float64
	
	// State for Physics
	currentEntropy  float64
	bidMass         float64 
	askMass         float64

	// Liquidation Tracking
	lastLiqVolume float64
	lastLiqSide   int8 // 1 for Buy, -1 for Sell

	// Whale Tracking (Major Players)
	whaleAccumulator float64
	whaleThreshold   float64 // USD value to be considered a whale trade
	
	// History Buffer (Keep for future)
	history []*FeatureVector
}

func (e *Engine) UpdateLiquidation(t core.Trade) {
	e.lastLiqVolume = t.Price * t.Size
	e.lastLiqSide = int8(t.Side)
}

func NewEngine(modelPath string) *Engine {
	var m *Model
	if modelPath != "" {
		var err error
		m, err = NewModel(modelPath)
		if err != nil {
			fmt.Printf("⚠️ Warning: Could not load ML model: %v\n", err)
		} else {
			fmt.Println("✅ ML model loaded successfully.")
		}
	}
	
	return &Engine{
		windowSize: 1 * time.Second, 
		entropy:    NewEntropyEngine(),
		model:      m,
		backtester: core.NewBacktester(281.25), // Initial: 9000 TWD (~281.25 USDT)
		history:    make([]*FeatureVector, 0, 100),
		whaleThreshold: 50000.0, // $50k USD for a single trade
	}
}

// GetBacktester returns the simulation engine
func (e *Engine) GetBacktester() *core.Backtester {
	return e.backtester
}

// UpdateTrade ingests a new trade and updates momentum features
func (e *Engine) UpdateTrade(t core.Trade) *FeatureVector {
	dt := float64(t.Timestamp - e.lastPriceTime) / 1e6 
	if dt <= 0 { return nil }

	// Kinematics
	dp := t.Price - e.lastPrice
	velocity := dp / dt
	dv := velocity - e.lastVelocity
	accel := dv / dt

	// Whale Detection (Major Player Volume)
	usdValue := t.Price * t.Size
	if usdValue >= e.whaleThreshold {
		sideFactor := 1.0
		if t.Side == core.Sell {
			sideFactor = -1.0
		}
		e.whaleAccumulator += usdValue * sideFactor
	}

	// Physics: Resistance
	currentMass := 1.0 
	if velocity > 0 {
		currentMass = math.Max(1.0, e.askMass)
	} else if velocity < 0 {
		currentMass = math.Max(1.0, e.bidMass)
	}
	
	e.lastPrice = t.Price
	e.lastPriceTime = t.Timestamp
	e.lastVelocity = velocity
	
	fv := &FeatureVector{
		Timestamp:      t.Timestamp,
		PriceVelocity:  velocity,
		PriceAccel:     accel,
		Entropy:        e.currentEntropy,
		MassResistance: currentMass,
		OrderImbalance: (e.bidMass - e.askMass) / math.Max(1, (e.bidMass + e.askMass)),
		NetForce:       e.lastLiqVolume * float64(e.lastLiqSide),
		WhaleForce:     e.whaleAccumulator,
	}
	
	// Reset after use (decay/reset per trade snapshot)
	e.lastLiqVolume = 0
	e.whaleAccumulator *= 0.95 // Exponential decay for WhaleForce to capture "pressure" rather than absolute sum
	
	// Maintain History
	e.history = append(e.history, fv)
	if len(e.history) > 100 {
		e.history = e.history[len(e.history)-100:]
	}
	
	return fv
}

// UpdateBook updates orderbook related features
func (e *Engine) UpdateBook(b *core.OrderBookL2) *FeatureVector {
	var bMass, aMass float64
	depth := 20
	if len(b.Bids) < depth { depth = len(b.Bids) }
	for i := 0; i < depth; i++ {
		bMass += b.Bids[i].Size
	}
	
	depth = 20
	if len(b.Asks) < depth { depth = len(b.Asks) }
	for i := 0; i < depth; i++ {
		aMass += b.Asks[i].Size
	}
	
	e.bidMass = bMass
	e.askMass = aMass

	ent := e.entropy.CalculateBookEntropy(b)
	e.currentEntropy = ent

	return &FeatureVector{
		Timestamp:      b.Timestamp,
		OrderImbalance: (bMass - aMass) / (bMass + aMass),
		Entropy:        ent,
		MassResistance: (bMass + aMass) / 2,
	}
}

// PredictFuture returns a Gaussian distribution for price at T + seconds
func (e *Engine) PredictFuture(seconds float64) Prediction {
	if e.lastPrice == 0 { return Prediction{} }

	// Try AI Prediction first
	if e.model != nil && e.model.available && len(e.history) >= 60 {
		// Flatten history to []float32
		// [velocity, accel, entropy, mass, imbalance, liq_force]
		input := make([]float32, 0, 60*6)
		
		// Use the last 60 frames
		frames := e.history[len(e.history)-60:]
		
		for _, v := range frames {
			input = append(input, float32(v.PriceVelocity))
			input = append(input, float32(v.PriceAccel))
			input = append(input, float32(v.Entropy))
			input = append(input, float32(v.MassResistance))
			input = append(input, float32(v.OrderImbalance))
			input = append(input, float32(v.NetForce))
		}
		
		// Predict with V3 (returns [Q10, Q50, Q90])
		// We use index 1 (Q50/Median) for our Mean prediction
		data := e.model.input.GetData()
		copy(data, input)
		err := e.model.session.Run()
		if err == nil {
			outData := e.model.output.GetData()
			predVal := outData[1] // Q50
			return Prediction{
				Mean: e.lastPrice * (1.0 + float64(predVal)),
				Sigma: 10.0, // Standard deviation placeholder
				Source: "AI V3",
			}
		}
	}

	// Heuristics
	v := e.lastVelocity
	damping := 1.0
	
	if v > 0 && e.askMass > 0 {
		damping = 1.0 / (1.0 + (e.askMass * 0.1)) 
	} else if v < 0 && e.bidMass > 0 {
		damping = 1.0 / (1.0 + (e.bidMass * 0.1))
	}
	
	effectiveV := v * damping
	mu := e.lastPrice + (effectiveV * seconds) 
	
	baseVol := e.lastPrice * 0.0005 
	entropyFactor := math.Max(0.5, e.currentEntropy - 2.5)
	
	sigma := baseVol * entropyFactor * math.Sqrt(seconds)

	return Prediction{
		Mean:  mu,
		Sigma: sigma,
		Source: "Heuristic",
	}
}
